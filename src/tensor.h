// 张量取用：把 GGUF 里的各种量化格式统一成"解到 float buffer"或"直接 GEMV"两种用法。
//
// 【为什么不统一解成 float】
// 2.44 GiB 的权重全解成 float 是 10 GiB，手机直接爆。
// 所以分两类：
//   - 小张量（RMSNorm、attn_gate）本来就是 F32，直接用
//   - 大张量（QKV/FFN/lm_head）走融合 GEMV，永不落地
#ifndef FSENGINE_TENSOR_H
#define FSENGINE_TENSOR_H

#include "gguf.h"
#include <stddef.h>

// 把一个张量当 float 向量读（只支持 F32/F16，给小权重用）
// out 必须能放下 n 个 float。返回 0 成功，非 0 失败。
int tensor_to_f32(GgufTensor *t, float *out, size_t n);

// 张量元素总数
size_t tensor_nelem(const GgufTensor *t);

// 一行（或一批行）用融合 GEMV 算 out[r] = dot(row_r, x)
// x 是 int8 激活，bs 是预计算的 bsums（Q4_K 用），可为 NULL（Q6_K 不需要）
// 支持 Q4_K / Q6_K / F32 / F16
void gemv_rows(GgufTensor *t, const int8_t *x, const int16_t *bs,
               float *out, size_t r0, size_t r1);

// 设置 GEMV 用几个线程（1 = 单线程直跑）
// 批量：X [n][cols] · W [rows][cols] -> Y [n][rows]
// bs 可为 NULL（Q6_K 不需要），布局 [n][cols/16+1]
void gemm_rows(GgufTensor *t, const int8_t *X, const int16_t *bs, float *Y,
               int n, size_t rows);

// per-block 激活量化版：as 是每 32 个一组的 float scale
void gemv_rows_b(GgufTensor *t, const int8_t *x, const float *as,
                 const int16_t *bs, float *out, size_t r0, size_t r1);

// x8 张量走 per-256 量化（传 float 激活），其余退回 per-32
void gemv_rows_f(GgufTensor *t, const float *xf,
                 const int8_t *qa, const float *as, const int16_t *bs,
                 float *out, size_t r0, size_t r1);

// 带 per-256 量化的版本（int8 缓存路径需要，粒度必须和权重一致）
// 用外部已准备好的 per-256 量化激活（避免每次 GEMV 重复量化同一个输入）
void gemv_rows_pre(GgufTensor *t, const int8_t *q8k, const float *as8k,
                   const int16_t *bs8k, const int8_t *qa, const float *as,
                   const int16_t *bs, float *out, size_t r0, size_t r1);

void gemv_rows_q8k(GgufTensor *t, const int8_t *q8k, const float *as8k,
                   const int8_t *qa, const float *as, const int16_t *bs,
                   float *out, size_t r0, size_t r1);

void gemv_set_threads(int n);
int  gemv_threads(void);

// x8 布局的批量 GEMV：一次读权重算 nb 个位置（nb<=4）
// out 按位置分段：[t0 的 r1-r0 个][t1 的 ...]...
void x8_compute_batch_impl(GgufTensor *t, const int8_t *x, const float *as,
                           const int16_t *bs, float *out, int nb,
                           size_t r0, size_t r1);
#endif
