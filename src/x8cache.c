#include "x8cache.h"
#include <sys/mman.h>
#include "repack.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 只重排这两个（占 50% 权重、每层都要读）
// ffn_down 不在列表里：它输入 10240 维，x8 的 per-256 量化开销淹没收益
static const char *kHot4[] = { "ffn_gate.weight", "ffn_up.weight",
                               "ffn_down.weight", "attn_output.weight" };
// Q6_K 的张量：attn_qkv（443MB）+ token_embd（262MB），占 33%
// Q6_K 的 x8：照 llama.cpp 的 arch/arm 实现重写后（用 vsli + bias 预计算），
// 比逐行快 1.16 倍（4.65 → 5.40 GB/s）。
// Q6_K 的 x8：单张量 1.16 倍，但接进引擎后无收益（试过复用外部量化也一样）。
// 原因：q6k_x8_block 每个 block 要展开 128 个 int16 scale（16 次 vst1q_s16），
// 而 Q4_K 的 x8 只解 4 组 —— 这部分开销抵消了 vdot 的收益。不启用。
static const char *kHot6[] = { };
// 环境变量控制只重排哪些（调试用）
static int hot_enabled(const char *name) {
    const char *e = getenv("X8_ONLY");
    if (!e) return 1;
    return strstr(name, e) != NULL;
}

typedef struct { char name[64]; void *data; size_t rowbytes; int type; } Entry;
static Entry *g_e = NULL;
static int g_n = 0, g_cap = 0;

const void *x8_lookup(const char *nm, size_t *rb) {
    for (int i = 0; i < g_n; i++)
        if (strcmp(g_e[i].name, nm) == 0) { if (rb) *rb = g_e[i].rowbytes; return g_e[i].data; }
    return NULL;
}

int x8_type(const char *nm) {
    for (int i = 0; i < g_n; i++) if (strcmp(g_e[i].name, nm) == 0) return g_e[i].type;
    return -1;
}

int x8_init(GgufModel *gm, int nthreads) {
    (void)nthreads;
    // x8（8 路交错）默认启用。
    // 关键：x8 内核要求激活按 per-256 量化（对应 llama.cpp 的 block_q8_K.d），
    // 所以 gate/up/down/attn_output 走 x8 时用 per-256，其余张量保持 per-32。
    // 两套量化并存，精度 8/8 token 与 llama.cpp 一致，速度 4.65 vs 4.13（+12.6%）。
    // 关掉用 X8_DISABLE=1。
    if (getenv("X8_DISABLE")) return 0;
    g_cap = 128;
    const int do_q6 = getenv("X8_Q6_OFF") == NULL;
    g_e = calloc(g_cap, sizeof(Entry));

    for (int t = 0; t < gm->n_tensors; t++) {
        GgufTensor *ts = &gm->tensors[t];
        int want4 = (ts->type == 12), want6 = (ts->type == 14);
        if (!want4 && !(want6 && do_q6)) continue;
        int hot = 0;
        const char **lst = want4 ? kHot4 : kHot6;
        const size_t nlst = want4 ? sizeof(kHot4)/sizeof(kHot4[0]) : sizeof(kHot6)/sizeof(kHot6[0]);
        for (size_t k = 0; k < nlst; k++)
            if (strstr(ts->name, lst[k])) { hot = 1; break; }
        if (!hot || !hot_enabled(ts->name)) continue;

        const size_t cols = ts->dims[0];
        const size_t rows = ts->dims[1];
        const size_t rb = want4 ? cols / 256 * 144 : cols / 256 * 210;
        if (rows % 8) continue;                    // 必须是 8 的倍数

        void *buf = malloc(rows * rb);
        if (!buf) { fprintf(stderr, "[x8] 内存不足，跳过 %s\n", ts->name); continue; }
        if (want4) repack_q4k_rows(ts->data, (int)rows, rb, buf);
        else       repack_q6k_rows(ts->data, (int)rows, rb, buf);

        if (g_n >= g_cap) { g_cap *= 2; g_e = realloc(g_e, g_cap * sizeof(Entry)); }
        snprintf(g_e[g_n].name, sizeof g_e[g_n].name, "%s", ts->name);
        g_e[g_n].data = buf;
        g_e[g_n].rowbytes = rb;
        g_e[g_n].type = ts->type;
        ts->x8_type = (int)ts->type;   // 标记到张量上，省掉后续每次查表
        ts->x8_ptr = buf;              // 指针也直接挂上，连 x8_lookup 都不用调
        ts->x8_rb  = rb;
        // 【释放原始页】重排后原布局不会再被读（有 x8 的走 x8 路径，
        // 没 x8 的走 gemv_rows_b），所以把原始 mmap 的页还给内核。
        // 不还的话模型 2488MB + x8 缓存 1431MB + KV = 3.9GB，直接挤爆内存
        // 开始用 swap，实测比不重排还慢 2 倍。
        if (getenv("X8_KEEP_ORIG") == NULL) {
            const size_t bytes = (size_t)rows * (size_t)(cols / 256) * (want4 ? 144 : 210);
            madvise((void *)ts->data, bytes, MADV_DONTNEED);
        }
        g_n++;
    }
    if (getenv("X8_VERBOSE")) {
        fprintf(stderr, "[x8] 重排 %d 个张量:\n", g_n);
        for (int i = 0; i < g_n && i < 40; i++)
            fprintf(stderr, "  %s rb=%zu\n", g_e[i].name, g_e[i].rowbytes);
    }
    return g_n;
}

int x8_has(const char *nm) {
    for (int i = 0; i < g_n; i++) if (strcmp(g_e[i].name, nm) == 0) return 1;
    return 0;
}

int x8_enabled(void) { return g_n > 0; }

void x8_free(void) {
    for (int i = 0; i < g_n; i++) free(g_e[i].data);
    free(g_e); g_e = NULL; g_n = g_cap = 0;
}
