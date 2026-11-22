// Tokenizer 的查找、预切分和 BPE merge。
#include "tok.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int tok_cmp(Tokenizer *tk, int a, int b) {
    const int la = tk->tok_len[a], lb = tk->tok_len[b];
    const int n = la < lb ? la : lb;
    const int c = memcmp(tk->tok_str[a], tk->tok_str[b], (size_t)n);
    if (c) return c;
    return la - lb;
}

int tok_lookup(Tokenizer *tk, const char *s, int len) {
    int lo = 0, hi = tk->n_tok - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        const int id = tk->tok_id[mid];
        const int l = tk->tok_len[id];
        const int n = l < len ? l : len;
        const int c = memcmp(tk->tok_str[id], s, (size_t)n);
        if (c == 0) {
            if (l == len) return id;
            if (l < len) lo = mid + 1; else hi = mid - 1;
        } else if (c < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

const char *tok_piece(Tokenizer *tk, int id, int *len) {
    if (id < 0 || id >= tk->n_tok) { *len = 0; return ""; }
    *len = tk->tok_len[id];
    return tk->tok_str[id];
}

// ── 预切分 ───────────────────────────────────────────────
// GPT-2 风格的切分：把文本切成「词」单元，每个单元内部才做 BPE merge。
// spark2_5 是自定义规则，这里按 Qwen/GPT-4 的通用做法：
//   1. 连续字母（含中文）归一段
//   2. 连续数字归一段
//   3. 连续标点归一段
//   4. 空白归一段
// 效果上接近，但和模型训练时的切分未必 100% 一致 —— 会体现在 token 数上。

static int is_letter(uint32_t cp) {
    // ASCII 字母 + 所有 >= 0x80 的码点（中文/日文/韩文/全角标点都算字母）
    // 这和 SentencePiece 的「非 ASCII 全算字母」一致，Qwen 系就是这么切的
    if (cp >= 'a' && cp <= 'z') return 1;
    if (cp >= 'A' && cp <= 'Z') return 1;
    if (cp >= 0x80) {
        // 但全角标点、CJK 标点要单独切出来
        if (cp >= 0x3000 && cp <= 0x303F) return 0;   // CJK 标点
        if (cp >= 0xFF00 && cp <= 0xFFEF) return 0;   // 全角标点
        return 1;
    }
    return 0;
}
static int is_digit(uint32_t cp) { return cp >= '0' && cp <= '9'; }
static int is_space(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x3000;
}

// 把一段字节做 BPE merge
static int bpe_merge(Tokenizer *tk, const char *s, int len, int *out, int max_out) {
    if (len <= 0) return 0;
    // 初始：每个字节一个符号（用 token id 表示，找不到就用 -1 表示未登录）
    int *sym = malloc(sizeof(int) * (size_t)len);
    int n = 0;
    for (int i = 0; i < len; i++) {
        const int id = tok_lookup(tk, s + i, 1);
        sym[n++] = (id >= 0) ? id : -1;
    }
    // 反复找「优先级最高（merge 序号最小）」的相邻对合并
    while (n > 1) {
        int best_rank = 1 << 30, best_at = -1;
        for (int i = 0; i < n - 1; i++) {
            if (sym[i] < 0 || sym[i+1] < 0) continue;
            // 拼出合并后的字节串再查表
            const int la = tk->tok_len[sym[i]], lb = tk->tok_len[sym[i+1]];
            char buf[256];
            if (la + lb >= (int)sizeof buf) continue;
            memcpy(buf, tk->tok_str[sym[i]], (size_t)la);
            memcpy(buf + la, tk->tok_str[sym[i+1]], (size_t)lb);
            const int mid = tok_lookup(tk, buf, la + lb);
            if (mid < 0) continue;
            // 用 merge 表找序号；没有就用一个很大的值
            int rank = 1 << 29;
            for (int m = 0; m < tk->n_merge; m++) {
                if (tk->merge_a[m] == sym[i] && tk->merge_b[m] == sym[i+1]) { rank = m; break; }
            }
            if (rank < best_rank) { best_rank = rank; best_at = i; }
        }
        if (best_at < 0) break;
        // 合并 best_at 和 best_at+1
        const int la = tk->tok_len[sym[best_at]], lb = tk->tok_len[sym[best_at+1]];
        char buf[256];
        if (la + lb >= (int)sizeof buf) break;
        memcpy(buf, tk->tok_str[sym[best_at]], (size_t)la);
        memcpy(buf + la, tk->tok_str[sym[best_at+1]], (size_t)lb);
        const int mid = tok_lookup(tk, buf, la + lb);
        if (mid < 0) break;
        sym[best_at] = mid;
        memmove(sym + best_at + 1, sym + best_at + 2, sizeof(int) * (size_t)(n - best_at - 2));
        n--;
    }
    int w = 0;
    for (int i = 0; i < n && w < max_out; i++) out[w++] = sym[i];
    free(sym);
    return w;
}

// 在 text 开头尝试匹配一个特殊 token（token_type != 1 且长度 >= 2）。
// 模板里的 <｜start▁of▁sentence｜>、<|Bot|> 这类标记必须整体识别成单 token，
// 否则会被 BPE 拆成 '<' 等碎片，模型就把它们当普通文本重复输出。
static int match_special(Tokenizer *tk, const char *p, int *len_out) {
    if (!tk->tok_type) return -1;
    int best = -1, best_len = 0;
    for (int i = 0; i < tk->n_tok; i++) {
        if (tk->tok_type[i] == 1) continue;
        const int l = tk->tok_len[i];
        if (l < 2 || l <= best_len) continue;
        if (!strncmp(p, tk->tok_str[i], (size_t)l)) { best = i; best_len = l; }
    }
    *len_out = best_len;
    return best;
}

int tok_encode(Tokenizer *tk, const char *text, int *out, int max_out) {
    int w = 0;
    const char *p = text;
    while (*p && w < max_out) {
        {
            int sl = 0;
            const int sid = match_special(tk, p, &sl);
            if (sid >= 0 && sl > 0) { out[w++] = sid; p += sl; continue; }
        }
        uint32_t cp;
        const char *q = p;
        if (!utf8_next(&q, &cp)) break;
        // 决定这一段有多长（同类的连续码点）
        int kind;
        if (is_space(cp)) kind = 3;
        else if (is_digit(cp)) kind = 2;
        else if (is_letter(cp)) kind = 1;
        else kind = 0;
        const char *s = p;
        const char *e = q;
        while (*e) {
            // 段内遇到特殊 token 就断开，别让 BPE 把 '<' 吃进普通 token
            // （之前 '.' + '<' 被合成 88343，导致后面的 <｜end▁of▁sentence｜> 认不出来）
            int sl0 = 0;
            if (match_special(tk, e, &sl0) >= 0 && sl0 > 0) break;
            uint32_t c2;
            const char *q2 = e;
            if (!utf8_next(&q2, &c2)) break;
            int k2;
            if (is_space(c2)) k2 = 3;
            else if (is_digit(c2)) k2 = 2;
            else if (is_letter(c2)) k2 = 1;
            else k2 = 0;
            if (k2 != kind) break;
            e = q2;
        }
        w += bpe_merge(tk, s, (int)(e - s), out + w, max_out - w);
        p = e;
    }
    return w;
}
