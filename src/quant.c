#include "quant.h"
#include <math.h>
#include <string.h>
#include <arm_neon.h>

// 【向量化】原来是纯标量：找 max 用逐元素比较，量化用逐元素 lrintf。
// 这两个函数每层要调 3~4 次（attn_norm 后、attn_out 前、ffn_norm 后、ffn_down 前），
// 尺寸 2560 或 10240，标量版是实打实的开销。

static inline float max_abs(const float *x, int n) {
    float32x4_t mx = vdupq_n_f32(0.f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const float32x4_t a0 = vabsq_f32(vld1q_f32(x + i));
        const float32x4_t a1 = vabsq_f32(vld1q_f32(x + i + 4));
        const float32x4_t a2 = vabsq_f32(vld1q_f32(x + i + 8));
        const float32x4_t a3 = vabsq_f32(vld1q_f32(x + i + 12));
        mx = vmaxq_f32(mx, vmaxq_f32(vmaxq_f32(a0, a1), vmaxq_f32(a2, a3)));
    }
    for (; i + 4 <= n; i += 4) mx = vmaxq_f32(mx, vabsq_f32(vld1q_f32(x + i)));
    float m = vmaxvq_f32(mx);
    for (; i < n; i++) { const float a = fabsf(x[i]); if (a > m) m = a; }
    return m;
}

// 把 n 个 float 按 inv 缩放、夹到 ±127、四舍五入成 int8
static inline void quantize_into(const float *x, int8_t *q, float inv, int n) {
    const float32x4_t vinv = vdupq_n_f32(inv);
    const float32x4_t vmax = vdupq_n_f32(127.f);
    const float32x4_t vmin = vdupq_n_f32(-127.f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const float32x4_t a0 = vmulq_f32(vld1q_f32(x + i),      vinv);
        const float32x4_t a1 = vmulq_f32(vld1q_f32(x + i + 4),  vinv);
        const float32x4_t a2 = vmulq_f32(vld1q_f32(x + i + 8),  vinv);
        const float32x4_t a3 = vmulq_f32(vld1q_f32(x + i + 12), vinv);
        const int32x4_t i0 = vcvtnq_s32_f32(vmaxq_f32(vminq_f32(a0, vmax), vmin));
        const int32x4_t i1 = vcvtnq_s32_f32(vmaxq_f32(vminq_f32(a1, vmax), vmin));
        const int32x4_t i2 = vcvtnq_s32_f32(vmaxq_f32(vminq_f32(a2, vmax), vmin));
        const int32x4_t i3 = vcvtnq_s32_f32(vmaxq_f32(vminq_f32(a3, vmax), vmin));
        const int16x8_t s01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
        const int16x8_t s23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
        vst1q_s8(q + i, vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23)));
    }
    for (; i < n; i++) {
        float v = x[i] * inv;
        v = v > 127.f ? 127.f : (v < -127.f ? -127.f : v);
        q[i] = (int8_t)lrintf(v);
    }
}

// 每 16 个 int8 求和（x8 内核用 bsums 修正无符号偏移）
static inline void bsums_into(const int8_t *q, int16_t *bs, int n) {
    for (int i = 0; i + 16 <= n; i += 16)
        bs[i / 16] = (int16_t)vaddlvq_s8(vld1q_s8(q + i));
}

void quant_act_block(const float *x, int8_t *q, float *scales, int16_t *bs, int n) {
    const int nb = n / 32;
    for (int b = 0; b < nb; b++) {
        const float *xs = x + b * 32;
        const float mx = max_abs(xs, 32);
        const float sc = (mx > 1e-8f) ? mx / 127.0f : 1.0f;
        scales[b] = sc;
        quantize_into(xs, q + b * 32, 1.0f / sc, 32);
    }
    if (bs) bsums_into(q, bs, n);
}

// per-256 量化（对齐 llama.cpp 的 block_q8_K）：
// 每 256 个元素一个 scale，而不是每 32 个。
// x8 内核要求这种粒度，否则整个 256-block 无法共用同一个 d。
void quant_act_q8k(const float *x, int8_t *q, float *d_out, int16_t *bs, int n) {
    const int nb = n / 256;
    for (int b = 0; b < nb; b++) {
        const float *xs = x + b * 256;
        int8_t *qs = q + b * 256;
        const float mx = max_abs(xs, 256);
        const float sc = (mx > 1e-8f) ? mx / 127.0f : 1.0f;
        d_out[b] = sc;
        quantize_into(xs, qs, 1.0f / sc, 256);
        if (bs) bsums_into(qs, bs + b * 16, 256);
    }
}
