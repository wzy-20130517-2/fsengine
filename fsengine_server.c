// fsengine-server —— 兼容 llama-server 的最小 HTTP 服务，供 ccmchat-cli 连接。
//
// 只实现 ccmchat-cli 实际用到的三个接口：
//   GET  /health                  → {"status":"ok"}
//   POST /completion              → SSE，字段 {content, stop}
//   POST /v1/chat/completions     → SSE，字段 {choices:[{delta:{content}}]}
//
// 【为什么自己写 HTTP】不引入任何依赖。Termux 上没有 libevent，
// 单用户场景也不需要并发，一个 accept 循环足够。
#include "engine.h"
#include "tok.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

static Engine   *g_e;
static Tokenizer g_tk;
static float    *g_logits;
static char      g_chat_tmpl[4096];
static int       g_vocab;
static int       g_eos = 1;       // tokenizer.ggml.eos_token_id
static int       g_tk_think = 3;  // <think>
static int       g_tk_ethink = 4; // </think>

// ── 采样 ────────────────────────────────────────────────
typedef struct {
    float temp, top_p, repeat_penalty;
    int   top_k, max_tokens;
    int  *hist;      // 已生成的 token，用于重复惩罚
    int   n_hist;
} Sampler;

static unsigned long g_rng = 12345;
static float frand(void) {
    g_rng = g_rng * 6364136223846793005UL + 1442695040888963407UL;
    return (float)((g_rng >> 33) & 0xFFFFFF) / 16777216.0f;
}

// 从 logits 挑一个 token。返回 -1 表示出错。
static int sample(float *lg, Sampler *s) {
    const int V = g_vocab;
    // 重复惩罚：出现过的 token 降权
    if (s->repeat_penalty > 1.0f && s->hist) {
        for (int i = 0; i < s->n_hist; i++) {
            const int t = s->hist[i];
            if (t < 0 || t >= V) continue;
            lg[t] = (lg[t] > 0) ? lg[t] / s->repeat_penalty : lg[t] * s->repeat_penalty;
        }
    }
    if (s->temp <= 0.0f) {          // greedy
        int best = 0;
        for (int i = 1; i < V; i++) if (lg[i] > lg[best]) best = i;
        return best;
    }
    for (int i = 0; i < V; i++) lg[i] /= s->temp;

    // top_k：只保留最大的 k 个。
    // 【别用「扫 k 遍找次大值」】那是 O(k×V)，k=40、V=131072 就是 524 万次
    // 比较，实测每 token 多花 400ms，把速度从 4.87 tok/s 拖到 1.53。
    // 这里用单遍分桶找阈值：把 [min,max] 切成 256 桶，数出每桶个数，
    // 从高往低累加到 k，确定阈值所在桶，再精扫一遍。O(V) 全程。
    if (s->top_k > 0 && s->top_k < V) {
        float lo = lg[0], hi = lg[0];
        for (int i = 1; i < V; i++) { if (lg[i] < lo) lo = lg[i]; if (lg[i] > hi) hi = lg[i]; }
        if (hi > lo) {
            enum { NB = 256 };
            int cnt[NB] = {0};
            const float scale = (float)(NB - 1) / (hi - lo);
            for (int i = 0; i < V; i++) {
                int b2 = (int)((lg[i] - lo) * scale);
                if (b2 < 0) b2 = 0; else if (b2 >= NB) b2 = NB - 1;
                cnt[b2]++;
            }
            int acc = 0, cut = NB - 1;
            for (; cut >= 0; cut--) {
                acc += cnt[cut];
                if (acc >= s->top_k) break;
            }
            // 阈值取 cut 桶的下边界，保证「>= 阈值」的不超过 k 太多
            const float thr = lo + (float)cut / scale;
            int kept = 0;
            for (int i = 0; i < V; i++) {
                if (lg[i] >= thr) kept++;
                else lg[i] = -1e30f;
            }
            // 分桶可能多留一些（同桶内无法区分），超出 4 倍就退化成按序裁剪
            if (kept > s->top_k * 4) {
                float t2 = thr;
                while (kept > s->top_k && t2 < hi) {
                    t2 += (hi - lo) / (float)(NB * 4);
                    kept = 0;
                    for (int i = 0; i < V; i++) if (lg[i] >= t2) kept++;
                }
                for (int i = 0; i < V; i++) if (lg[i] < t2) lg[i] = -1e30f;
            }
        }
    }
    // softmax + top_p
    float mx = -1e30f;
    for (int i = 0; i < V; i++) if (lg[i] > mx) mx = lg[i];
    double sum = 0;
    for (int i = 0; i < V; i++) { lg[i] = expf(lg[i] - mx); sum += lg[i]; }
    if (sum <= 0) return 0;
    for (int i = 0; i < V; i++) lg[i] = (float)(lg[i] / sum);

    if (s->top_p > 0.0f && s->top_p < 1.0f) {
        // 按概率降序累加，超过 top_p 的截断。为了不排序，用阈值法近似：
        // 先算 top_p 对应的概率阈值（用二分）
        float lo = 0, hi = 1;
        for (int it = 0; it < 20; it++) {
            const float mid = (lo + hi) * 0.5f;
            double acc = 0;
            for (int i = 0; i < V; i++) if (lg[i] >= mid) acc += lg[i];
            if (acc > s->top_p) lo = mid; else hi = mid;
        }
        for (int i = 0; i < V; i++) if (lg[i] < lo) lg[i] = 0;
        double s2 = 0;
        for (int i = 0; i < V; i++) s2 += lg[i];
        if (s2 > 0) for (int i = 0; i < V; i++) lg[i] = (float)(lg[i] / s2);
    }
    const float r = frand();
    double acc = 0;
    for (int i = 0; i < V; i++) { acc += lg[i]; if (acc >= r) return i; }
    return V - 1;
}

// ── JSON 小工具（够用就行，不引第三方）──────────────────
// 从 JSON 里取字符串字段，处理 \" \\ \n \uXXXX
static int json_str(const char *js, const char *key, char *out, int cap) {
    char pat[64]; snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(js, pat);
    if (!p) return 0;
    p = strchr(p + strlen(pat), ':');
    if (!p) return 0;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '"') return 0;
    p++;
    int w = 0;
    while (*p && *p != '"' && w < cap - 1) {
        if (*p == '\\') {
            p++;
            switch (*p) {
            case 'n': out[w++] = '\n'; p++; break;
            case 't': out[w++] = '\t'; p++; break;
            case 'r': out[w++] = '\r'; p++; break;
            case '"': out[w++] = '"';  p++; break;
            case '\\': out[w++] = '\\'; p++; break;
            case '/': out[w++] = '/';  p++; break;
            case 'u': {
                unsigned cp = 0;
                for (int i = 0; i < 4 && p[1]; i++) {
                    p++;
                    const char h = *p;
                    cp <<= 4;
                    if (h >= '0' && h <= '9') cp |= (unsigned)(h - '0');
                    else if (h >= 'a' && h <= 'f') cp |= (unsigned)(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') cp |= (unsigned)(h - 'A' + 10);
                }
                p++;
                // UTF-8 编码
                if (cp < 0x80) out[w++] = (char)cp;
                else if (cp < 0x800) { out[w++] = (char)(0xC0 | (cp >> 6)); out[w++] = (char)(0x80 | (cp & 0x3F)); }
                else { out[w++] = (char)(0xE0 | (cp >> 12)); out[w++] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[w++] = (char)(0x80 | (cp & 0x3F)); }
                break;
            }
            default: out[w++] = *p; p++; break;
            }
        } else out[w++] = *p++;
    }
    out[w] = 0;
    return w;
}

static float json_num(const char *js, const char *key, float def) {
    char pat[64]; snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(js, pat);
    if (!p) return def;
    p = strchr(p + strlen(pat), ':');
    if (!p) return def;
    return (float)atof(p + 1);
}

// 把 messages 数组拼成 ChatML 风格的 prompt（按 Spark 的模板）
// 按 Spark 的 chat template 拼 prompt。
// 格式（从 tokenizer.chat_template 读出来）：
//   <｜start▁of▁sentence｜><|System|>...<｜end▁of▁sentence｜>
//   <｜start▁of▁sentence｜><|User|>...<｜end▁of▁sentence｜>
//   <｜start▁of▁sentence｜><|Bot|><think>        （思考开）
//   <｜start▁of▁sentence｜><|Bot|></think>       （思考关）
//
// 【不能只 strstr("role") 然后找 "content"】多轮时 content 会串到下一个
// message 里。这里按 messages 数组逐对象解析，每个对象内单独找 role/content。
static void build_chat_prompt(const char *body, char *out, int cap, int enable_thinking) {
    int w = 0;
    const char *arr = strstr(body, "\"messages\"");
    if (!arr) { snprintf(out, (size_t)cap, "<|Bot|><think>"); return; }
    const char *p = strchr(arr, '[');
    if (!p) p = arr;
    else p++;

    // 【必须有 system 块】Spark 的模板里 system 块是无条件输出的：
    //   <｜start▁of▁sentence｜><|System|>\n{default_system}<｜end▁of▁sentence｜>
    // default_system = "you are a helpful assistant."
    // 之前用户没传 system 就整个跳过，导致 prompt 结构和训练时不一致 ——
    // 这是 enable_thinking=false 放 </think> 也不生效的原因。
    char sys_text[8192] = {0};
    int have_system = 0;
    {
        const char *r = strstr(body, "\"role\"");
        while (r && r < p) {
            char role0[32] = {0};
            json_str(r, "role", role0, sizeof role0);
            if (!strcmp(role0, "system")) {
                json_str(r, "content", sys_text, sizeof sys_text);
                have_system = 1; break;
            }
            r = strstr(r + 6, "\"role\"");
        }
    }
    // 【system 内容不能自己编】模板里 default_system 只由 reasoning_instructions
    // 组成，用户没传 system、思考又关闭时它就是空的。硬塞 "you are a helpful
    // assistant." 等于给模型加了一句它训练时没有的提示词。
    w += snprintf(out + w, (size_t)(cap - w),
                  "<｜start▁of▁sentence｜><|System|>\n%s<｜end▁of▁sentence｜>", sys_text);
    (void)have_system;

    while (*p && w < cap - 64) {
        // 找到下一个对象
        const char *obj_start = strchr(p, '{');
        if (!obj_start) break;
        const char *obj_end = strchr(obj_start, '}');
        if (!obj_end) break;

        // 在这个对象范围内取 role 和 content
        char obj[16384];
        size_t olen = (size_t)(obj_end - obj_start) + 1;
        if (olen >= sizeof obj) olen = sizeof obj - 1;
        memcpy(obj, obj_start, olen);
        obj[olen] = 0;

        char role[32] = {0}, content[12288] = {0};
        json_str(obj, "role", role, sizeof role);
        json_str(obj, "content", content, sizeof content);

        const char *tag = NULL;
        if (!strcmp(role, "system")) {
            // 已有 system 块，跳过（模板里只有第一条 system 进初始块）
            p = obj_end + 1;
            continue;
        }
        else if (!strcmp(role, "user")) tag = "User";
        else if (!strcmp(role, "assistant")) tag = "Bot";
        if (tag) {
            w += snprintf(out + w, (size_t)(cap - w),
                          "<｜start▁of▁sentence｜><|%s|>%s<｜end▁of▁sentence｜>",
                          tag, content);
        }
        p = obj_end + 1;
    }

    // 生成提示：思考开就放 <think>，关就直接放 </think>（告诉模型跳过思考段）
    // 照模板：思考开放 <think>，思考关放 </think>
    snprintf(out + w, (size_t)(cap - w), "<｜start▁of▁sentence｜><|Bot|>%s",
             enable_thinking ? "<think>" : "</think>");
}

// ── 生成 + SSE ─────────────────────────────────────────
static int write_all(int fd, const char *p, size_t n) {
    while (n) {
        const ssize_t k = write(fd, p, n);
        if (k <= 0) return 0;
        p += k; n -= (size_t)k;
    }
    return 1;
}

// 判断是否是停止串（只看已生成文本末尾）
static int ends_with(const char *s, int slen, const char *pat) {
    const int pl = (int)strlen(pat);
    if (pl <= 0 || slen < pl) return 0;
    return !memcmp(s + slen - pl, pat, (size_t)pl);
}

static void sse_chunk(int fd, const char *json) {
    char buf[16384];
    const int n = snprintf(buf, sizeof buf, "data: %s\n\n", json);
    write_all(fd, buf, (size_t)(n > 0 ? n : 0));
}

// 生成主循环。openai=1 用 chat.completions 的 chunk 格式，否则用 /completion
static void generate(int fd, const char *prompt, Sampler *sp, int openai,
                     const char *stop1, const char *stop2, int enable_thinking) {
    fprintf(stderr, "[srv] TOKENIZE start len=%zu\n", strlen(prompt));
    int ids[8192];
    if (getenv("SRV_DUMP")) {
        fprintf(stderr, "[dump] len=%zu: ", strlen(prompt));
        for (const char *q = prompt; *q; q++) {
            if (*q == '\n') fprintf(stderr, "\\n");
            else if (*q == '\t') fprintf(stderr, "\\t");
            else fputc(*q, stderr);
        }
        fprintf(stderr, "\n");
    }
    const int np = tok_encode(&g_tk, prompt, ids, 8192);
    fprintf(stderr, "[srv] TOKENIZE done np=%d\n", np);
    fprintf(stderr, "[srv] prompt %d tokens, max %d\n", np, sp->max_tokens);

    if (np <= 0) { fprintf(stderr, "[srv] 空 prompt\n"); return; }
    if (np >= g_e->max_ctx - sp->max_tokens) {
        fprintf(stderr, "[srv] prompt %d + max %d 超过上下文 %d，截断 max_tokens\n", np, sp->max_tokens, g_e->max_ctx);
        sp->max_tokens = g_e->max_ctx - np - 1;
        if (sp->max_tokens < 1) sp->max_tokens = 1;
    }
    fprintf(stderr, "[srv] IDS %d %d %d %d\n", np>0?ids[0]:-1, np>1?ids[1]:-1, np>2?ids[2]:-1, np>3?ids[3]:-1);
    float *lg = NULL;
    struct timespec pf0, pf1; clock_gettime(CLOCK_MONOTONIC, &pf0);
    for (int i = 0; i < np; i++) {
        if (ids[i] < 0 || ids[i] >= g_vocab) { fprintf(stderr, "[srv] 非法 token id %d @%d\n", ids[i], i); return; }
        lg = engine_step(g_e, ids[i]);
        if (!lg) { fprintf(stderr, "[srv] engine_step 返回 NULL @%d\n", i); return; }
    }
    clock_gettime(CLOCK_MONOTONIC, &pf1);
    const double pre_ms = ((pf1.tv_sec - pf0.tv_sec) * 1000.0) + ((pf1.tv_nsec - pf0.tv_nsec) / 1e6);
    fprintf(stderr, "[srv] PREFILL done %.0fms\n", pre_ms);
    if (!lg) return;

    sp->hist = malloc(sizeof(int) * (size_t)sp->max_tokens);
    sp->n_hist = 0;

    char text[16384]; int tlen = 0;
    // 思考开时 prompt 末尾是 <think>，从第一个 token 起就在思考段；
    // 思考关时没有思考段，所有输出都算正文。
    int in_think = enable_thinking;
    struct timespec ts_start; clock_gettime(CLOCK_MONOTONIC, &ts_start);
    int ngen = 0;

    for (int step = 0; step < sp->max_tokens; step++) {
        if (step == 0) fprintf(stderr, "[gen] 进入采样, V=%d temp=%.2f top_k=%d top_p=%.2f\n", g_vocab, sp->temp, sp->top_k, sp->top_p);
        const int tok = sample(lg, sp);
        if (step == 0) fprintf(stderr, "[gen] 采样得到 %d\n", tok);
        if (getenv("SRV_DBG")) fprintf(stderr, "  [gen] step=%d tok=%d\n", step, tok);
        if (tok < 0) break;
        if (sp->n_hist < sp->max_tokens) sp->hist[sp->n_hist++] = tok;

        int plen = 0;
        const char *piece = tok_piece(&g_tk, tok, &plen);
        // EOS 直接停，不输出
        if (tok == g_eos) break;
        // prompt 里已经带了 <think>，所以从第一个 token 开始就是思考内容。
        // 遇到 </think>(id 4) 才切到正文。之前把 </think> 当普通文本输出了。
        if (tok == g_tk_ethink) {
            in_think = 0;
            lg = engine_step(g_e, tok);
            if (!lg) break;
            continue;
        }
        if (plen > 0 && tlen + plen < (int)sizeof text) {
            memcpy(text + tlen, piece, (size_t)plen);
            tlen += plen;
            text[tlen] = 0;

            // 输出这一片（JSON 转义）
            char esc[4096]; int w = 0;
            for (int i = 0; i < plen && w < (int)sizeof esc - 8; i++) {
                const unsigned char ch = (unsigned char)piece[i];
                if (ch == '"' || ch == '\\') { esc[w++] = '\\'; esc[w++] = (char)ch; }
                else if (ch == '\n') { esc[w++] = '\\'; esc[w++] = 'n'; }
                else if (ch == '\r') { esc[w++] = '\\'; esc[w++] = 'r'; }
                else if (ch == '\t') { esc[w++] = '\\'; esc[w++] = 't'; }
                else if (ch < 0x20) { w += snprintf(esc + w, sizeof esc - (size_t)w, "\\u%04x", ch); }
                else esc[w++] = (char)ch;
            }
            esc[w] = 0;

            char json[8192];
            if (openai)
                snprintf(json, sizeof json,
                    "{\"choices\":[{\"delta\":{\"%s\":\"%s\"},\"index\":0,\"finish_reason\":null}]}",
                    in_think ? "reasoning_content" : "content", esc);
            else
                snprintf(json, sizeof json, "{\"content\":\"%s\"}", esc);
            sse_chunk(fd, json);
        }
        ngen++;

        if (ends_with(text, tlen, stop1) || (stop2 && ends_with(text, tlen, stop2))) break;

        struct timespec p0, p1; clock_gettime(CLOCK_MONOTONIC, &p0);
        lg = engine_step(g_e, tok);
        clock_gettime(CLOCK_MONOTONIC, &p1);
        const double ts = p0.tv_sec + p0.tv_nsec / 1e9, te = p1.tv_sec + p1.tv_nsec / 1e9;
        if (getenv("SRV_PROF") && step < 5)
            fprintf(stderr, "[prof] step=%d engine_step=%.0fms\n",
                    step, (te - ts) * 1000);
        if (!lg) break;
    }
    struct timespec ts_end; clock_gettime(CLOCK_MONOTONIC, &ts_end);
    const double dt = (ts_end.tv_sec - ts_start.tv_sec) + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    // 收尾：带 timings 的空 chunk + [DONE]
    char fin[512];
    if (openai)
        snprintf(fin, sizeof fin,
            "{\"choices\":[{\"delta\":{},\"index\":0,\"finish_reason\":\"stop\"}],"
            "\"timings\":{\"predicted_n\":%d,\"predicted_ms\":%.1f,"
            "\"predicted_per_second\":%.2f,\"prompt_n\":%d,\"prompt_ms\":%.1f,"
            "\"prompt_per_second\":%.2f}}",
            ngen, dt * 1000, ngen / (dt > 0.001 ? dt : 0.001),
            np, pre_ms, np / (pre_ms > 0.001 ? pre_ms / 1000 : 0.001));
    else
        snprintf(fin, sizeof fin,
            "{\"content\":\"\",\"stop\":true,\"timings\":{\"predicted_n\":%d,\"predicted_ms\":%.1f,"
            "\"predicted_per_second\":%.2f,\"prompt_n\":%d,\"prompt_ms\":%.1f,"
            "\"prompt_per_second\":%.2f}}",
            ngen, dt * 1000, ngen / (dt > 0.001 ? dt : 0.001),
            np, pre_ms, np / (pre_ms > 0.001 ? pre_ms / 1000 : 0.001));
    sse_chunk(fd, fin);
    write_all(fd, "data: [DONE]\n\n", 15);
    free(sp->hist); sp->hist = NULL;
}

// ── HTTP 主循环 ─────────────────────────────────────────
static int read_request(int fd, char *buf, int cap) {
    int n = 0;
    while (n < cap - 1) {
        const ssize_t k = recv(fd, buf + n, (size_t)(cap - 1 - n), 0);
        if (k <= 0) break;
        n += (int)k;
        buf[n] = 0;
        // 找到 body 结束
        char *hdr_end = strstr(buf, "\r\n\r\n");
        if (hdr_end) {
            char *cl = strcasestr(buf, "content-length:");
            if (cl) {
                const int want = (int)(hdr_end + 4 - buf) + atoi(cl + 15);
                if (n >= want) break;
            } else break;
        }
    }
    return n;
}

static void respond(int fd, int code, const char *ctype, const char *body) {
    char hdr[512];
    const int n = snprintf(hdr, sizeof hdr,
        "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
        code, code == 200 ? "OK" : "Error", ctype, strlen(body));
    write_all(fd, hdr, (size_t)n);
    write_all(fd, body, strlen(body));
}

// 【绑大核】这台机器是 4×A77(2.6GHz) + 4×A55(2.0GHz)，
// A55 的 NEON/SDOT 吞吐只有 A77 的 1/3。不绑核时线程池的 worker 会被
// 调度到小核上，实测比绑核慢 2.7 倍（730ms vs 270ms）。
// bionic 没有 pthread_setaffinity_np，直接用 sched_setaffinity 系统调用。
#include <sched.h>
#include <sys/syscall.h>
static void bind_big_cores(void) {
    // bionic 的 cpu_set_t 只有 CPU_SETSIZE=32 位但宏不完整，直接用 64 位掩码。
    // cpu4-7 对应 bit 4..7，即 0xF0。
    const unsigned long mask = 0xF0UL;
    const long r = syscall(SYS_sched_setaffinity, 0, sizeof(mask), &mask);
    if (r != 0) fprintf(stderr, "[srv] 绑核失败（继续跑，但会慢 2-3 倍）\n");
    else        fprintf(stderr, "[srv] 已绑定 cpu4-7（大核）\n");
}

int main(int argc, char **argv) {
    bind_big_cores();
    // stderr 重定向到文件时是全缓冲，崩溃会丢掉最后几行 —— 调试时必须关掉
    setvbuf(stderr, NULL, _IONBF, 0);
    setvbuf(stdout, NULL, _IONBF, 0);
    // 参数：兼容 llama-server 的 -m/-c/-t。-m 指定模型，位置参数也接。
    const char *model_path = NULL;
    int ctx_len = 512, n_threads = 3, port = 8931;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc)      model_path = argv[++i];
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) ctx_len = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--host") || !strcmp(argv[i], "--no-webui")) { if (argv[i][0]=='-' && argv[i][1]=='-' && !strcmp(argv[i],"--host")) i++; }
        else if (argv[i][0] != '-' && !model_path)       model_path = argv[i];
        else if (argv[i][0] != '-' && atoi(argv[i]) > 0)  port = atoi(argv[i]);
    }
    if (!model_path) { fprintf(stderr, "用法: %s -m 模型.gguf [-c 上下文] [-t 线程] [--port 端口]\n", argv[0]); return 1; }
    if (ctx_len < 64) ctx_len = 64;
    if (n_threads < 1) n_threads = 1;
    fprintf(stderr, "[srv] 模型=%s ctx=%d t=%d port=%d\n", model_path, ctx_len, n_threads, port);

    fprintf(stderr, "[srv] 加载分词器...\n");
    if (!tok_load(model_path, &g_tk)) { fprintf(stderr, "分词器加载失败\n"); return 1; }

    fprintf(stderr, "[srv] 加载引擎...\n");
    g_e = engine_open_t(model_path, ctx_len, n_threads);   // 2048 会分配 300MB KV，把模型挤出 page cache
    if (!g_e) { fprintf(stderr, "引擎加载失败\n"); return 1; }
    g_vocab = g_e->V;
    g_logits = g_e->logits;

    const int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket"); return 1; }
    int on = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &on, sizeof on);
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = htons((uint16_t)port);
    if (bind(srv, (struct sockaddr *)&a, sizeof a) < 0) { perror("bind"); return 1; }
    if (listen(srv, 8) < 0) { perror("listen"); return 1; }
    fprintf(stderr, "[srv] 监听 127.0.0.1:%d\n", port);
    if (getenv("SRV_BENCH")) {
        fprintf(stderr, "[bench] 引擎自测（pos 0..5，模拟 bench_stable2）:\n");
        // 先热身 6 轮，把模型页读进 page cache
        for (int w = 0; w < 6; w++) { g_e->pos = 0; for (int k = 0; k < 6; k++) engine_step(g_e, k); }
        fprintf(stderr, "[bench] 热身完成\n");
        for (int r = 0; r < 4; r++) {
            if (r) usleep(2000000);   // 和 bench_stable2 一样冷却 2 秒
            struct timespec ts0, ts1;
            clock_gettime(CLOCK_MONOTONIC, &ts0);
            g_e->pos = 0;
            for (int k = 0; k < 6; k++) engine_step(g_e, k);
            clock_gettime(CLOCK_MONOTONIC, &ts1);
            const double dt = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec) / 1e9;
            fprintf(stderr, "[bench] 第%d轮 %.0f ms/token = %.2f tok/s\n",
                    r, dt / 6 * 1000, 6 / dt);
        }
    }

    static char req[1 << 20];
    for (;;) {
        const int fd = accept(srv, NULL, NULL);
        if (fd < 0) continue;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof on);
        const int n = read_request(fd, req, (int)sizeof req);
        if (n <= 0) { close(fd); continue; }

        if (!strncmp(req, "GET /health", 11)) {
            respond(fd, 200, "application/json", "{\"status\":\"ok\"}");
            close(fd);
            continue;
        }
        if (!strncmp(req, "OPTIONS", 7)) {
            respond(fd, 200, "text/plain", "");
            close(fd);
            continue;
        }
        // 找 body
        char *body = strstr(req, "\r\n\r\n");
        body = body ? body + 4 : req;

        Sampler sp = { .temp = json_num(body, "temperature", 0.7f),
                       .top_p = json_num(body, "top_p", 0.9f),
                       .repeat_penalty = json_num(body, "repeat_penalty", 1.1f),
                       .top_k = (int)json_num(body, "top_k", 40),
                       .max_tokens = (int)json_num(body, "max_tokens",
                                       json_num(body, "n_predict", 512)) };
        if (sp.max_tokens <= 0 || sp.max_tokens > 4096) sp.max_tokens = 512;

        const int openai = !strncmp(req, "POST /v1/chat/completions", 25);

        char hdr[512];
        const int hn = snprintf(hdr, sizeof hdr,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\nConnection: close\r\n"
            "Access-Control-Allow-Origin: *\r\n\r\n");
        write_all(fd, hdr, (size_t)hn);

        char prompt[16384];
        int think = 1;
        {
            const char *et = strstr(body, "\"enable_thinking\"");
            if (et) { const char *vv = strchr(et + 17, ':'); if (vv) { vv++; while (*vv==' '||*vv=='\t') vv++; think = (*vv != 'f'); } }
        }
        if (openai) {
            // chat_template_kwargs.enable_thinking 控制思考开关，默认开
            build_chat_prompt(body, prompt, sizeof prompt, think);
        } else {
            json_str(body, "prompt", prompt, sizeof prompt);
        }
        fprintf(stderr, "[srv] %s, prompt=%zu 字节\n", openai ? "chat" : "completion", strlen(prompt));
        generate(fd, prompt, &sp, openai, "</answer>", "<｜end▁of▁sentence｜>", think);
        close(fd);
    }
    return 0;
}
