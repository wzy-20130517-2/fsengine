#include "tensor.h"
#include "gguf.h"
#include "pool.h"
#include "x8cache.h"
#include "i8cache.h"
#include "repack.h"
#include <string.h>
#include <math.h>
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>

// 全局：当前用几线程跑 GEMV（0 = 单线程直跑）
static int g_nw = 1;
void gemv_set_threads(int n) { g_nw = n < 1 ? 1 : n; }
int  gemv_threads(void) { return g_nw; }

// int8 缓存的 GEMV：权重已经是 int8，直接 vdot。
// 实测比 Q6_K 原路径快 1.95 倍（4.74 → 11.26 GB/s）。
static void i8_range(const int8_t *w8, const float *ws, size_t cols,
                     const int8_t *x, const float *as,
                     float *out, size_t r0, size_t r1) {
    const size_t nb = cols / 256;
    // 权重是 per-256 量化，激活也得按 per-256 才能严格配对。
    // per-32 的 as 只能取平均，会掉精度（实测第 3 步开始错）。
    // 这里直接用 as 的块首值 —— 因为 as 是这个 256-block 内的 8 个值，
    // 它们本来就来自同一个 256 区间，首值即为该区间第一个 32 组的 scale。
    // 为了严格对齐，改成把激活也按 per-256 重新量化。
    for (size_t r = r0; r < r1; r++) {
        const int8_t *wr = w8 + r * cols;
        const float *sr = ws + r * nb;
        float total = 0.f;
        for (size_t b = 0; b < nb; b++) {
            int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
            const int8_t *wb = wr + b * 256;
            const int8_t *xb = x + b * 256;
            for (int i = 0; i < 256; i += 32) {
                a0 = vdotq_s32(a0, vld1q_s8(wb + i),      vld1q_s8(xb + i));
                a1 = vdotq_s32(a1, vld1q_s8(wb + i + 16), vld1q_s8(xb + i + 16));
            }
            const int32_t dot = vaddvq_s32(vaddq_s32(a0, a1));
            // as8k 已经是 per-256（每 256 个一个），直接按 b 取
            total += (float)dot * sr[b] * as[b];
        }
        out[r] = total;
    }
}

typedef struct {
    const int8_t *w8; const float *ws; size_t cols;
    const int8_t *x; const float *as;
    float *out; size_t r0, r1; int nw;
} I8Task;

static void i8_worker(void *arg, int wid) {
    I8Task *k = (I8Task *)arg;
    const size_t total = k->r1 - k->r0;
    size_t per = (total + k->nw - 1) / k->nw;
    if (!per) per = total;
    size_t a = k->r0 + (size_t)wid * per;
    if (a >= k->r1) return;
    size_t b = a + per > k->r1 ? k->r1 : a + per;
    i8_range(k->w8, k->ws, k->cols, k->x, k->as, k->out, a, b);
}



// ── x8 路径：8 行交错权重 ────────────────────────────
// 一次算 8 行，权重读一次、激活复制 8 份。
// 实测比逐行 GEMV 快 3.09x（3.14 → 9.70 GB/s）。
extern void q4k_x8_block(const blk_q4Kx8*, const int8_t*, const float*, const int16_t*, float*);
extern void q6k_x8_block(const void*, const int8_t*, const float*, float*);


// x8 路径：内部按 per-256 重新量化激活（x8 内核要求整块共用一个 scale）。
// 这样外部只需传原始 float 激活，不用维护两套量化缓冲。
// x8 的量化部分独立出来（只做一次，然后多线程共享）
static void x8_prepare(const float *xf, size_t cols,
                       int8_t **xq_o, int16_t **bq_o, float **dq_o) {
    const size_t nblk = cols / 256;
    static int8_t *xq = NULL; static int16_t *bq = NULL; static float *dq = NULL;
    static size_t cap = 0;
    if (cols > cap) {
        free(xq); free(bq); free(dq);
        xq = malloc(cols);
        bq = malloc((cols / 16 + 1) * 2);
        dq = malloc((cols / 256 + 1) * sizeof(float));
        cap = cols;
    }
    // 【向量化】原来是纯标量：找 max 用 256 次比较、量化用 256 次 lrintf。
    // 每层 gate/up 各调一次（10240 维 = 40 block），标量版占了可观的时间。
    const float32x4_t vmax = vdupq_n_f32(127.f);
    const float32x4_t vmin = vdupq_n_f32(-127.f);
    for (size_t b = 0; b < nblk; b++) {
        const float *src = xf + b * 256;
        int8_t *dst = xq + b * 256;
        float32x4_t mxv = vdupq_n_f32(0.f);
        for (int i = 0; i < 256; i += 4) {
            const float32x4_t v = vld1q_f32(src + i);
            mxv = vmaxq_f32(mxv, vabsq_f32(v));
        }
        const float mx = vmaxvq_f32(mxv);
        const float sc = (mx > 1e-8f) ? mx / 127.f : 1.f;
        dq[b] = sc;
        const float32x4_t vinv = vdupq_n_f32(1.f / sc);
        for (int i = 0; i < 256; i += 16) {
            const float32x4_t a0 = vmulq_f32(vld1q_f32(src + i),      vinv);
            const float32x4_t a1 = vmulq_f32(vld1q_f32(src + i + 4),  vinv);
            const float32x4_t a2 = vmulq_f32(vld1q_f32(src + i + 8),  vinv);
            const float32x4_t a3 = vmulq_f32(vld1q_f32(src + i + 12), vinv);
            const int32x4_t i0 = vcvtnq_s32_f32(vmaxq_f32(vminq_f32(a0, vmax), vmin));
            const int32x4_t i1 = vcvtnq_s32_f32(vmaxq_f32(vminq_f32(a1, vmax), vmin));
            const int32x4_t i2 = vcvtnq_s32_f32(vmaxq_f32(vminq_f32(a2, vmax), vmin));
            const int32x4_t i3 = vcvtnq_s32_f32(vmaxq_f32(vminq_f32(a3, vmax), vmin));
            const int16x8_t s01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
            const int16x8_t s23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
            vst1q_s8(dst + i, vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23)));
        }
        for (int i = 0; i < 16; i++) {
            const int8x16_t v = vld1q_s8(dst + i * 16);
            bq[b * 16 + i] = (int16_t)vaddlvq_s8(v);
        }
    }
    *xq_o = xq; *bq_o = bq; *dq_o = dq;
}

// 纯计算部分（不含量化），供线程池调用
// 【交错分配】nw 个线程按 g = wid + i*nw 轮流取 8 行组，而不是各切一大段。
// 这样多个线程同时读的地址挨在一起，DRAM bank 冲突更少。
static void x8_compute_stride(const void *packed, size_t rowbytes, size_t cols, int qtype,
                       const int8_t *xq, const int16_t *bq, const float *dq,
                       float *out, size_t r0, size_t r1, size_t stride, size_t phase) {
    const size_t nblk = cols / 256;
    const size_t blk_sz = (qtype == 14) ? 1680 : 1152;   // Q6_Kx8 vs Q4_Kx8
    // 【分支提到循环外】原来每个 block 都判断一次 qtype，
    // 1280 个行组 × 10 个 block = 12800 次分支。提到外面整个循环只判一次。
    if (qtype == 14) {
        size_t r = r0 + phase * 8;
        for (; r + 8 <= r1; r += 8 * stride) {
            const uint8_t *base = (const uint8_t *)packed + r * rowbytes;
            float acc[8];
            for (size_t b = 0; b < nblk; b++) {
                float o8[8];
                q6k_x8_block(base + b * blk_sz, xq + b * 256, dq + b, o8);
                if (b == 0) for (int j = 0; j < 8; j++) acc[j] = o8[j];
                else        for (int j = 0; j < 8; j++) acc[j] += o8[j];
            }
            for (int j = 0; j < 8; j++) out[r + j] = acc[j];
        }
        for (; r < r1; r++) out[r] = 0.f;
        return;
    }
    size_t r = r0 + phase * 8;
    for (; r + 8 <= r1; r += 8 * stride) {
        const uint8_t *base = (const uint8_t *)packed + r * rowbytes;
        float acc[8];
        for (size_t b = 0; b < nblk; b++) {
            float o8[8];
            q4k_x8_block((const blk_q4Kx8 *)(base + b * blk_sz),
                         xq + b * 256, dq + b, bq + b * 16, o8);
            if (b == 0) for (int j = 0; j < 8; j++) acc[j] = o8[j];
            else        for (int j = 0; j < 8; j++) acc[j] += o8[j];
        }
        for (int j = 0; j < 8; j++) out[r + j] = acc[j];
    }
    for (; r < r1; r++) out[r] = 0.f;
}

typedef struct {
    const void *packed; size_t rowbytes, cols; int qtype;
    const int8_t *xq; const int16_t *bq; const float *dq;
    float *out; size_t r0, r1; int nw;
} X8FTask;

static void x8f_worker(void *arg, int wid) {
    X8FTask *k = (X8FTask *)arg;
    {
        // 每线程切一段连续行。交错分配（g = wid + i*nw）实测更慢：
        // 破坏了硬件预取器，1.07ms vs 0.91ms。
        const size_t ngrp = (k->r1 - k->r0) / 8;
        size_t gper = (ngrp + k->nw - 1) / k->nw;
        if (!gper) gper = 1;
        size_t ga = (size_t)wid * gper;
        if (ga >= ngrp) return;
        size_t gb = ga + gper > ngrp ? ngrp : ga + gper;
        if (gb <= ga) return;
        x8_compute_stride(k->packed, k->rowbytes, k->cols, k->qtype, k->xq, k->bq, k->dq,
                          k->out, k->r0 + ga * 8, k->r0 + gb * 8, 1, 0);
    }
}

static void x8_range_f(const void *packed, size_t rowbytes, size_t cols, int qt,
                       const float *xf, float *out, size_t r0, size_t r1) {
    // 1) 量化一次（per-256）
    int8_t *xq; int16_t *bq; float *dq;
    x8_prepare(xf, cols, &xq, &bq, &dq);

    // 2) 计算：多线程按 8 行组切分
    const int nw = g_nw;
    if (nw <= 1) {
        x8_compute_stride(packed, rowbytes, cols, qt, xq, bq, dq, out, r0, r1, 1, 0);
    } else {
        X8FTask k = { packed, rowbytes, cols, qt, xq, bq, dq, out, r0, r1, nw };
        PoolJob j = { x8f_worker, &k };
        pool_run(&j, nw);
    }
}

static void x8_range(const void *packed, size_t rowbytes, size_t cols,
                     const int8_t *x, const float *as, const int16_t *bs,
                     float *out, size_t r0, size_t r1) {
    const size_t nblk = cols / 256;
    size_t r = r0;
    for (; r + 8 <= r1; r += 8) {
        const blk_q4Kx8 *base = (const blk_q4Kx8 *)((const uint8_t *)packed + r * rowbytes);
        float acc[8];
        for (size_t b = 0; b < nblk; b++) {
            float o8[8];
            // 【per-256 scale】x8 内核要求整个 256-block 用一个激活 scale
            // （对应 llama.cpp 的 block_q8_K.d）。这里把 per-32 的 8 个值
            // 取平均作为该 block 的 scale —— 因为 8 个值来自同一个 256 区间，
            // 平均后误差可接受（实测 8/8 token 与 llama.cpp 一致）。
            // 直接传 per-32 的 8 个 scale，内核内部按 sb 组取平均
            q4k_x8_block(base + b, x + b * 256, as + b * 8, bs ? bs + b * 16 : NULL, o8);
            if (b == 0) for (int j = 0; j < 8; j++) acc[j] = o8[j];
            else        for (int j = 0; j < 8; j++) acc[j] += o8[j];
        }
        for (int j = 0; j < 8; j++) out[r + j] = acc[j];
    }
    // 尾部不足 8 行（缓存里存的是原样，走不了 x8）—— 调用方保证 rows%8==0
    for (; r < r1; r++) out[r] = 0.f;
}

typedef struct {
    const void *packed; size_t rowbytes, cols;
    const int8_t *x; const float *as; const int16_t *bs;
    float *out; size_t r0, r1; int nw;
} X8Task;

static void x8_worker(void *arg, int wid) {
    X8Task *k = (X8Task *)arg;
    // 【切分必须严格按 8 对齐且不重叠】
    // 原来 per 先向上取整到 8 的倍数，a 再对齐一次 —— 会跳过一些行（漏算）。
    // 正确做法：per 先除以 8 取整，a/b 直接用 8 的倍数。
    const size_t ngrp = (k->r1 - k->r0) / 8;     // 总共有多少个 8 行组
    size_t gper = (ngrp + k->nw - 1) / k->nw;    // 每组线程分几个 8 行组
    if (!gper) gper = 1;
    size_t ga = (size_t)wid * gper;
    if (ga >= ngrp) return;
    size_t gb = ga + gper > ngrp ? ngrp : ga + gper;
    if (gb <= ga) return;
    x8_range(k->packed, k->rowbytes, k->cols, k->x, k->as, k->bs, k->out,
             k->r0 + ga * 8, k->r0 + gb * 8);
}



float q4k_dot_bs(const void *raw, const int8_t *x, const int16_t *bs, size_t n);
float q6k_dot(const void *raw, const int8_t *x, size_t n);
float q2k_dot(const void *raw, const int8_t *x, size_t n);
float q3k_dot(const void *raw, const int8_t *x, size_t n);
float q5k_dot_bs(const void *raw, const int8_t *x, const int16_t *bs, size_t n);
void q4k_gemm_row(const void *raw, const int8_t *X, const int16_t *bs,
                  float *Y, int n, size_t cols, size_t ldY);
void q4k_gemm_row_a(const void *raw, const int8_t *X, const int16_t *bs,
                    const float *asc, float *Y, int n, size_t cols, size_t ldY);
void q6k_gemm_row_a(const void *raw, const int8_t *X, const float *asc, float *Y,
                    int n, size_t cols, size_t ldY);
void q6k_gemm_row(const void *raw, const int8_t *X, float *Y,
                  int n, size_t cols, size_t ldY);
void q5k_gemm_row(const void *raw, const int8_t *X, const int16_t *bs,
                  float *Y, int n, size_t cols, size_t ldY);
void dequant_q4k_row(const void *raw, float *out, size_t n);
void dequant_q6k_row(const void *raw, float *out, size_t n);

// fp16 -> float（tensor.c 自己一份，不依赖 gemv.c 的内部表）
static inline float h2f(uint16_t h) {
    __fp16 v; memcpy(&v, &h, 2);
    return (float)v;
}

size_t tensor_nelem(const GgufTensor *t) {
    size_t n = 1;
    for (int i = 0; i < t->n_dims; i++) n *= t->dims[i];
    return n;
}

int tensor_to_f32(GgufTensor *t, float *out, size_t n) {
    if (!t || !t->data) return -1;
    if (t->type == 0) {                       // F32
        memcpy(out, t->data, n * sizeof(float));
        return 0;
    }
    if (t->type == 1) {                       // F16
        const uint16_t *p = (const uint16_t *)t->data;
        for (size_t i = 0; i < n; i++) out[i] = h2f(p[i]);
        return 0;
    }
    return -2;                                // 量化类型不该走这条
}

// 一行 float 点积（F32/F16 权重用）
static float dot_f32(const void *row, int type, const int8_t *x, size_t n) {
    float s = 0.f;
    if (type == 0) {
        const float *w = (const float *)row;
        for (size_t i = 0; i < n; i++) s += w[i] * (float)x[i];
    } else {
        const uint16_t *w = (const uint16_t *)row;
        for (size_t i = 0; i < n; i++) s += h2f(w[i]) * (float)x[i];
    }
    return s;
}

static void gemv_rows_range(GgufTensor *t, const int8_t *x, const int16_t *bs,
                            float *out, size_t r0, size_t r1) {
    const size_t cols = t->dims[0];
    size_t rb;
    switch (t->type) {
    case 10: rb = cols / 256 * 84;  break;    // Q2_K
    case 11: rb = cols / 256 * 110; break;    // Q3_K
    case 12: rb = cols / 256 * 144; break;    // Q4_K
    case 13: rb = cols / 256 * 176; break;    // Q5_K
    case 14: rb = cols / 256 * 210; break;    // Q6_K
    case 0:  rb = cols * 4; break;            // F32
    case 1:  rb = cols * 2; break;            // F16
    default: rb = 0;
    }
    if (!rb) return;

    // 【switch 提到循环外】逐行判断类型等于每行一次分支，10240 行就是 10240 次。
    switch (t->type) {
    case 10: for (size_t r = r0; r < r1; r++) out[r] = q2k_dot(t->data + r * rb, x, cols); break;
    case 11: for (size_t r = r0; r < r1; r++) out[r] = q3k_dot(t->data + r * rb, x, cols); break;
    case 12: for (size_t r = r0; r < r1; r++) out[r] = q4k_dot_bs(t->data + r * rb, x, bs, cols); break;
    case 13: for (size_t r = r0; r < r1; r++) out[r] = q5k_dot_bs(t->data + r * rb, x, bs, cols); break;
    case 14: for (size_t r = r0; r < r1; r++) out[r] = q6k_dot(t->data + r * rb, x, cols); break;
    default: for (size_t r = r0; r < r1; r++) out[r] = dot_f32(t->data + r * rb, t->type, x, cols); break;
    }
}

// 带线程池的版本：按输出行切分到各 worker
typedef struct {
    GgufTensor    *t;
    const int8_t  *x;
    const int16_t *bs;
    float         *out;
    size_t         r0, r1;   // 绝对行号区间
    int            nw;
} GemvTask;

static void gemv_worker(void *arg, int wid) {
    GemvTask *k = (GemvTask *)arg;
    const size_t total = k->r1 - k->r0;
    size_t per = (total + k->nw - 1) / k->nw;
    if (!per) per = total;
    size_t a = k->r0 + (size_t)wid * per;
    if (a >= k->r1) return;
    size_t b = a + per > k->r1 ? k->r1 : a + per;
    gemv_rows_range(k->t, k->x, k->bs, k->out, a, b);
}

void gemv_rows(GgufTensor *t, const int8_t *x, const int16_t *bs,
               float *out, size_t r0, size_t r1) {
    if (!t || r1 <= r0) return;
    const int nw = g_nw;
    // 行数太少时并行不划算 —— 线程池同步本身有开销
    if (nw <= 1 || (r1 - r0) < (size_t)nw * 8) {
        gemv_rows_range(t, x, bs, out, r0, r1);
        return;
    }
    GemvTask k = { t, x, bs, out, r0, r1, nw };
    PoolJob j = { gemv_worker, &k };
    pool_run(&j, nw);
}

// 批量：一次算 n 个 token × [r0,r1) 行
// X 布局 [n][cols]，bs 布局 [n][cols/16+1]，Y 布局 [n][rows]
// asc 非空时是 per-32 激活 scale，布局 [n][cols/32+1]，此时结果为最终值
typedef struct {
    GgufTensor *t;
    const int8_t *X;
    const int16_t *bs;
    const float  *asc;
    float *Y;
    int n;
    size_t r0, r1, rows;
    int nw;
} GemmTask;

static void gemm_worker(void *arg, int wid) {
    GemmTask *k = (GemmTask *)arg;
    if (getenv("DBG_GEMM")) fprintf(stderr, "[gemm] wid=%d n=%d rows=%zu type=%u\n", wid, k->n, k->rows, k->t->type);
    const size_t total = k->r1 - k->r0;
    size_t per = (total + k->nw - 1) / k->nw;
    if (!per) per = total;
    size_t a = k->r0 + (size_t)wid * per;
    if (a >= k->r1) return;
    size_t b = a + per > k->r1 ? k->r1 : a + per;

    const size_t cols = k->t->dims[0];
    size_t bs_bytes;
    switch (k->t->type) {
    case 10: bs_bytes = 84;  break;
    case 11: bs_bytes = 110; break;
    case 12: bs_bytes = 144; break;
    case 13: bs_bytes = 176; break;
    case 14: bs_bytes = 210; break;
    default: bs_bytes = 144;
    }
    const size_t rb = cols / 256 * bs_bytes;

    for (size_t r = a; r < b; r++) {
        const void *row = k->t->data + r * rb;
        // 每个 worker 写自己那几行，用临时缓冲再散到 Y 的对应列
        float tmp[16];
        if (k->n > 16) return;      // 目前 batch 上限 16，够投机解码用
        if (k->t->type == 14) {
            q6k_gemm_row_a(row, k->X, k->asc, tmp, k->n, cols, 1);
        } else if (k->t->type == 13) {
            q5k_gemm_row(row, k->X, k->bs, tmp, k->n, cols, 1);
        } else if (k->t->type == 12) {
            q4k_gemm_row_a(row, k->X, k->bs, k->asc, tmp, k->n, cols, 1);
        } else {
            for (int t = 0; t < k->n; t++)
                tmp[t] = (k->t->type == 10)
                       ? q2k_dot(row, k->X + (size_t)t * cols, cols)
                       : q3k_dot(row, k->X + (size_t)t * cols, cols);
        }
        for (int t = 0; t < k->n; t++) k->Y[(size_t)t * k->rows + r] = tmp[t];
    }
}

void gemm_rows_a(GgufTensor *t, const int8_t *X, const int16_t *bs,
                 const float *asc, float *Y, int n, size_t rows) {
    if (!t || n <= 0 || rows == 0) return;
    const int nw = g_nw;
    GemmTask k = { t, X, bs, asc, Y, n, 0, rows, rows, nw };
    if (nw <= 1) { gemm_worker(&k, 0); return; }
    PoolJob j = { gemm_worker, &k };
    pool_run(&j, nw);
}

void gemm_rows(GgufTensor *t, const int8_t *X, const int16_t *bs, float *Y,
               int n, size_t rows) {
    gemm_rows_a(t, X, bs, NULL, Y, n, rows);
}

// per-block 激活量化版的 GEMV：as 是每 32 个一组的 scale
extern float q4k_dot_bsb(const void*, const int8_t*, const float*, const int16_t*, size_t);
extern float q5k_dot_bsb(const void*, const int8_t*, const float*, const int16_t*, size_t);
extern float q6k_dot_b(const void*, const int8_t*, const float*, size_t);
extern float q2k_dot_b(const void*, const int8_t*, const float*, size_t);
extern float q2k_dot_bb(const void*, const int8_t*, const float*, const int16_t*, size_t);
extern float q3k_dot_b(const void*, const int8_t*, const float*, size_t);


static void gemv_rows_b_range(GgufTensor *t, const int8_t *x, const float *as,
                             const int16_t *bs, float *out, size_t r0, size_t r1) {
    const size_t cols = t->dims[0];
    size_t rb;
    switch (t->type) {
    case 10: rb = cols / 256 * 84;  break;
    case 11: rb = cols / 256 * 110; break;
    case 12: rb = cols / 256 * 144; break;
    case 13: rb = cols / 256 * 176; break;
    case 14: rb = cols / 256 * 210; break;
    case 0:  rb = cols * 4; break;
    case 1:  rb = cols * 2; break;
    default: rb = 0;
    }
    if (!rb) return;

    switch (t->type) {
    case 10:
        for (size_t r = r0; r < r1; r++) out[r] = q2k_dot_bb(t->data + r * rb, x, as, bs, cols);
        break;
    case 11:
        for (size_t r = r0; r < r1; r++) out[r] = q3k_dot_b(t->data + r * rb, x, as, cols);
        break;
    case 12: {
        // 【4 行展开】一次算 4 行，让 CPU 能交错调度 4 条独立的解码流水线。
        // 实测比逐行快 1.19x（bench_mega D vs A）——
        // 单行的解码依赖链够长，4 行交错后能填满流水线气泡。
        size_t r = r0;
        for (; r + 4 <= r1; r += 4) {
            const uint8_t *b0 = (const uint8_t *)t->data + r * rb;
            out[r]     = q4k_dot_bsb(b0,            x, as, bs, cols);
            out[r + 1] = q4k_dot_bsb(b0 + rb,       x, as, bs, cols);
            out[r + 2] = q4k_dot_bsb(b0 + 2 * rb,   x, as, bs, cols);
            out[r + 3] = q4k_dot_bsb(b0 + 3 * rb,   x, as, bs, cols);
        }
        for (; r < r1; r++)
            out[r] = q4k_dot_bsb((const uint8_t *)t->data + r * rb, x, as, bs, cols);
        break;
    }
    case 13:
        for (size_t r = r0; r < r1; r++) out[r] = q5k_dot_bsb(t->data + r * rb, x, as, bs, cols);
        break;
    case 14:
        for (size_t r = r0; r < r1; r++) out[r] = q6k_dot_b(t->data + r * rb, x, as, cols);
        break;
    default: {
        const int nb32 = (int)(cols / 32);
        const size_t esz = (t->type == 0) ? 4 : 2;
        for (size_t r = r0; r < r1; r++) {
            const uint8_t *row = (const uint8_t *)t->data + r * rb;
            float acc = 0.f;
            for (int bi = 0; bi < nb32; bi++)
                acc += dot_f32(row + (size_t)bi * 32 * esz, t->type, x + bi * 32, 32) * as[bi];
            out[r] = acc;
        }
        break;
    }
    }
}
typedef struct {
    GgufTensor    *t;
    const int8_t  *x;
    const float   *as;
    const int16_t *bs;
    float         *out;
    size_t         r0, r1;
    int            nw;
} GemvTaskB;

static void gemv_b_worker(void *arg, int wid) {
    GemvTaskB *k = (GemvTaskB *)arg;
    const size_t total = k->r1 - k->r0;
    size_t per = (total + k->nw - 1) / k->nw;
    if (!per) per = total;
    size_t a = k->r0 + (size_t)wid * per;
    if (a >= k->r1) return;
    size_t b = a + per > k->r1 ? k->r1 : a + per;
    gemv_rows_b_range(k->t, k->x, k->as, k->bs, k->out, a, b);
}

void gemv_rows_b(GgufTensor *t, const int8_t *x, const float *as,
                 const int16_t *bs, float *out, size_t r0, size_t r1) {
    if (!t || r1 <= r0) return;

    // 【只走普通路径】x8 快路径全部由 gemv_rows_f 负责（它需要 float 激活）。
    // 这里如果也查 x8 缓存，会把重排后的数据当成原始布局读 —— 踩过。
    const int nw = g_nw;
    if (nw <= 1) {
        gemv_rows_b_range(t, x, as, bs, out, r0, r1);
        return;
    }
    GemvTaskB k = { t, x, as, bs, out, r0, r1, nw };
    PoolJob j = { gemv_b_worker, &k };
    pool_run(&j, nw);
}



// 面向 x8 张量的入口：传 float 激活，内部按 per-256 量化。
// 非 x8 张量自动退回普通路径（用调用方给的 per-32 量化结果）。
void gemv_rows_f(GgufTensor *t, const float *xf,
                 const int8_t *qa, const float *as, const int16_t *bs,
                 float *out, size_t r0, size_t r1) {
    if (!t || r1 <= r0) return;
    // 用张量上的标记位快速判断，避免每次都做线性查找 + strcmp
    if (!t->x8_type) { gemv_rows_b(t, qa, as, bs, out, r0, r1); return; }
    const size_t xrb = t->x8_rb;
    const void *xp = t->x8_ptr;
    const int qt = t->x8_type;
    if (xp) {
        x8_range_f(xp, xrb, t->dims[0], qt, xf, out, r0, r1);
        return;
    }
    gemv_rows_b(t, qa, as, bs, out, r0, r1);
}


// 面向 int8 缓存的入口：传 per-256 量化的激活（和权重量化粒度严格对齐）。
// 非 int8 缓存的张量退回普通路径。
void gemv_rows_q8k(GgufTensor *t, const int8_t *q8k, const float *as8k,
                   const int8_t *qa, const float *as, const int16_t *bs,
                   float *out, size_t r0, size_t r1) {
    if (!t || r1 <= r0) return;
    const float *ws = NULL; size_t icols = 0;
    const int8_t *w8 = i8_lookup(t->name, &ws, &icols);
    if (w8 && icols == t->dims[0]) {
        const int nw = g_nw;
        if (nw <= 1) {
            i8_range(w8, ws, icols, q8k, as8k, out, r0, r1);
        } else {
            I8Task k = { w8, ws, icols, q8k, as8k, out, r0, r1, nw };
            PoolJob j = { i8_worker, &k };
            pool_run(&j, nw);
        }
        return;
    }
    gemv_rows_b(t, qa, as, bs, out, r0, r1);
}


// 用外部准备好的 per-256 量化激活。
// ffn_gate 和 ffn_up 输入相同（都是 e->h），原来各自调 x8_prepare
// 量化两遍 2560 个元素；这个入口让调用方只量化一次。
void gemv_rows_pre(GgufTensor *t, const int8_t *q8k, const float *as8k,
                   const int16_t *bs8k, const int8_t *qa, const float *as,
                   const int16_t *bs, float *out, size_t r0, size_t r1) {
    if (!t || r1 <= r0) return;
    if (!t->x8_type) { gemv_rows_b(t, qa, as, bs, out, r0, r1); return; }
    const size_t xrb = t->x8_rb;
    const void *xp = t->x8_ptr;
    const int qt = t->x8_type;
    if (xp) {
        const int nw = g_nw;
        if (nw <= 1) {
            x8_compute_stride(xp, xrb, t->dims[0], qt, q8k, bs8k, as8k, out, r0, r1, 1, 0);
        } else {
            X8FTask k = { xp, xrb, t->dims[0], qt, q8k, bs8k, as8k, out, r0, r1, nw };
            PoolJob j = { x8f_worker, &k };
            pool_run(&j, nw);
        }
        return;
    }
    gemv_rows_b(t, qa, as, bs, out, r0, r1);
}

// ═══════════════════════════════════════════════════════════
// x8 布局上的批量 GEMV：一次读权重，同时算 nb 个位置。
//
// 【为什么重要】单位置 x8 内核实测 11.72 GB/s（3 线程），
// 而 batch=2 相当于每位置 23.44 GB/s —— 权重只读一次算两次。
// 原来的批量路径走 gemm_packed（原始 Q4_K 布局，按行），只有 4.96 GB/s。
//
// xq/as/bs 的布局：每个位置 256 字节激活 + 1 个 scale + 16 个 bsums，
// 位置之间按 256 / 1 / 16 的 stride 排列（由调用方保证）。
typedef struct {
    const void *packed; size_t rowbytes; size_t cols; int qtype;
    const int8_t *x; const float *as; const int16_t *bs;
    float *out; size_t r0, r1; int nw, nb;
} X8BTask;

static void x8b_worker(void *arg, int wid) {
    X8BTask *k = (X8BTask *)arg;
    const size_t ngrp = (k->r1 - k->r0) / 8;
    const int nw = k->nw > 1 ? k->nw : 1;
    size_t gper = (ngrp + nw - 1) / nw;
    if (!gper) gper = 1;
    const size_t ga = (size_t)wid * gper;
    if (ga >= ngrp) return;
    size_t gb = ga + gper > ngrp ? ngrp : ga + gper;
    if (gb <= ga) return;

    const size_t cols = k->cols, nblk = cols / 256;
    const size_t blk_sz = (k->qtype == 14) ? 1680 : 1152;
    const int nb = k->nb;

    float stack_out[4 * 8];
    for (size_t g = ga; g < gb; g++) {
        const uint8_t *base = (const uint8_t *)k->packed + (k->r0 + g * 8) * k->rowbytes;
        float acc[4 * 8];
        for (size_t b = 0; b < nblk; b++) {
            if (k->qtype == 14) {
                // Q6_K 暂不支持 batch（单位置版也没启用），退回逐位置
                for (int t = 0; t < nb; t++) {
                    float o8[8];
                    q6k_x8_block(base + b * blk_sz,
                                 k->x + (size_t)t * cols + b * 256,
                                 k->as + (size_t)t * nblk + b, o8);
                    if (b == 0) for (int j = 0; j < 8; j++) acc[t*8+j] = o8[j];
                    else        for (int j = 0; j < 8; j++) acc[t*8+j] += o8[j];
                }
            } else {
                // 激活布局：x[t*cols + b*256]，as[t*nblk + b]，bs[t*16..] 取当前 block 的 16 个
                // as 布局是 [位置][block]，取 block b 时要按位置步进 nblk
                // bs 布局是 [位置][16]，直接传头指针，内核内部按 t*16 索引
                {
                    float asc[4]; int16_t bsc[4 * 16];
                    for (int t = 0; t < nb; t++) {
                        asc[t] = k->as[(size_t)t * nblk + b];
                        // 位置 t 的 block b 对应 16 个 bsums
                        memcpy(bsc + t * 16,
                               k->bs + (size_t)t * (nblk * 16) + b * 16,
                               16 * sizeof(int16_t));
                    }
                    q4k_x8_block_n((const blk_q4Kx8 *)(base + b * blk_sz),
                                   k->x + b * 256, cols, asc, bsc, stack_out, nb);
                }
                if (b == 0) for (int j = 0; j < nb * 8; j++) acc[j] = stack_out[j];
                else        for (int j = 0; j < nb * 8; j++) acc[j] += stack_out[j];
            }
        }
        for (int t = 0; t < nb; t++)
            for (int j = 0; j < 8; j++) k->out[(size_t)t * (k->r1 - k->r0) + (g * 8 + j)] = acc[t*8+j];
    }
}

void x8_compute_batch_impl(GgufTensor *t, const int8_t *x, const float *as,
                           const int16_t *bs, float *out, int nb,
                           size_t r0, size_t r1) {
    if (!t || !t->x8_ptr || r1 <= r0) return;
    const int nw = gemv_threads();
    X8BTask k = { t->x8_ptr, t->x8_rb, t->dims[0], t->x8_type,
                  x, as, bs, out, r0, r1, nw, nb };
    if (nw <= 1) { x8b_worker(&k, 0); return; }
    PoolJob j = { x8b_worker, &k };
    pool_run(&j, nw);
}
