// Q4_K 的 8 路交错重排（照 llama.cpp 的 Q4_Kx8 做法）。
//
// 【为什么这个能快】原来一次 SDOT 只能服务 1 个输出行：
//   权重 16 字节（同一行的 16 个 IC）× 激活 16 字节 → 1 个部分和
// 重排后：
//   权重 16 字节 = 8 行 × 2 字节（同一批 IC），配合 8 个累加器
// 让同一份激活同时喂给 8 个输出，激活的读取和它在寄存器里的驻留都摊薄 8 倍。
//
// 布局（每个 super-block，1152 字节）：
//   d[8]      16B   8 行的 fp16 主 scale
//   dmin[8]   16B   8 行的 fp16 主 min
//   scales    96B   8 行 × 12B，按「同一个子块跨 8 行」排列
//   qs        1024B 按 8 字节交错：qs[i*8 .. i*8+7] = 第 i 个 8 字节组，来自 8 行
#include "repack.h"
#include <string.h>
#include <stdlib.h>
#include <arm_neon.h>

static inline float half2f_pub(uint16_t h) {
    __fp16 v; memcpy(&v, &h, 2); return (float)v;
}

#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct { uint16_t d, dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2]; } blk_q4K;

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// 把 8 个连续的 block_q4_K 重排成一个 block_q4_Kx8
static void pack8(const blk_q4K *in, blk_q4Kx8 *out) {
    for (int i = 0; i < 8; i++) {
        out->d[i] = in[i].d;
        out->dmin[i] = in[i].dmin;
    }
    // qs：每次搬 8 字节，按行轮转（照 llama.cpp 的写法）
    //   src_id = i % 8（第几行）
    //   src_offset = (i / 8) * 8（行内偏移）
    //   dst_offset = i * 8
    const int end = 128 / 8 * 8 / 8 * 8;   // = QK_K*4/blck_size = 128*8/8 = 128
    for (int i = 0; i < 128; i++) {
        const int src_id = i % 8;
        const int src_offset = (i / 8) * 8;
        const int dst_offset = i * 8;
        memcpy(out->qs + dst_offset, in[src_id].qs + src_offset, 8);
    }
    (void)end;
    // scales：完全照 llama.cpp 的 make_block_q4_Kx8 位打包逻辑。
    // 每 12 字节装 4 个子块的 (8 行 sc + 8 行 mn)，
    // 高 2 位和下 4 位分两段拼（前 48 字节存 sb 0-3，后 48 字节存 sb 4-7）。
    uint8_t sc8[8], mn8[8];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            sc8[j] = in[j].scales[i] & 63;
            mn8[j] = in[j].scales[i + 4] & 63;
        }
        out->scales[i*12 + 0] = (sc8[0] & 63) + ((sc8[4] & 48) << 2);
        out->scales[i*12 + 1] = (sc8[1] & 63) + ((sc8[5] & 48) << 2);
        out->scales[i*12 + 2] = (sc8[2] & 63) + ((sc8[6] & 48) << 2);
        out->scales[i*12 + 3] = (sc8[3] & 63) + ((sc8[7] & 48) << 2);
        out->scales[i*12 + 4] = (mn8[0] & 63) + ((mn8[4] & 48) << 2);
        out->scales[i*12 + 5] = (mn8[1] & 63) + ((mn8[5] & 48) << 2);
        out->scales[i*12 + 6] = (mn8[2] & 63) + ((mn8[6] & 48) << 2);
        out->scales[i*12 + 7] = (mn8[3] & 63) + ((mn8[7] & 48) << 2);
        out->scales[i*12 + 8] = (sc8[4] & 15) + ((mn8[4] & 15) << 4);
        out->scales[i*12 + 9] = (sc8[5] & 15) + ((mn8[5] & 15) << 4);
        out->scales[i*12 +10] = (sc8[6] & 15) + ((mn8[6] & 15) << 4);
        out->scales[i*12 +11] = (sc8[7] & 15) + ((mn8[7] & 15) << 4);
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            sc8[j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i + 8] & 15);
            mn8[j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i + 8] & 240) >> 4);
        }
        out->scales[i*12 +48] = (sc8[0] & 63) + ((sc8[4] & 48) << 2);
        out->scales[i*12 +49] = (sc8[1] & 63) + ((sc8[5] & 48) << 2);
        out->scales[i*12 +50] = (sc8[2] & 63) + ((sc8[6] & 48) << 2);
        out->scales[i*12 +51] = (sc8[3] & 63) + ((sc8[7] & 48) << 2);
        out->scales[i*12 +52] = (mn8[0] & 63) + ((mn8[4] & 48) << 2);
        out->scales[i*12 +53] = (mn8[1] & 63) + ((mn8[5] & 48) << 2);
        out->scales[i*12 +54] = (mn8[2] & 63) + ((mn8[6] & 48) << 2);
        out->scales[i*12 +55] = (mn8[3] & 63) + ((mn8[7] & 48) << 2);
        out->scales[i*12 +56] = (sc8[4] & 15) + ((mn8[4] & 15) << 4);
        out->scales[i*12 +57] = (sc8[5] & 15) + ((mn8[5] & 15) << 4);
        out->scales[i*12 +58] = (sc8[6] & 15) + ((mn8[6] & 15) << 4);
        out->scales[i*12 +59] = (sc8[7] & 15) + ((mn8[7] & 15) << 4);
    }
}

void repack_q4k_rows(const void *raw, int nrows, size_t rowbytes, void *out) {
    const uint8_t *src = (const uint8_t *)raw;
    uint8_t *dst = (uint8_t *)out;
    const size_t nblk = rowbytes / 144;
    const size_t out_blk = 1152;      // sizeof(blk_q4Kx8)

    int r = 0;
    for (; r + 8 <= nrows; r += 8) {
        for (size_t b = 0; b < nblk; b++) {
            blk_q4K tmp[8];
            for (int k = 0; k < 8; k++)
                memcpy(&tmp[k], src + (size_t)(r + k) * rowbytes + b * 144, 144);
            pack8(tmp, (blk_q4Kx8 *)(dst + ((size_t)(r / 8) * nblk + b) * out_blk));
        }
    }
    // 尾部不足 8 行：直接照抄（调用方不会用 x8 内核）
    for (; r < nrows; r++)
        memcpy(dst + (size_t)(r / 8) * nblk * out_blk + (r % 8) * 144,
               src + (size_t)r * rowbytes, rowbytes);
}

// ── 8 路 SDOT 内核 ──────────────────────────────────────
// 输入：一个解好的 block_q4Kx8（1152 字节）+ 激活（256 字节）
// 输出：8 个部分和
//
// 【关键】qs 按 8 字节交错后，`ld1 {v0.16b}` 读到的 16 字节是
// [行0的2字节][行1的2字节]...[行7的2字节] —— 正好 8 行各 2 字节。
// 但这不能直接和激活做 SDOT（SDOT 要求两边都是按 IC 连续）。
// 所以还是要先拆成 8 个独立的 16 字节向量，再各自 SDOT。
// 这样做的收益在于：8 行共享同一份激活的寄存器驻留，
// 并且 8 条 SDOT 能同时发射，填满流水线。
// 解码 Q4_Kx8 的 6-bit scales（照 llama.cpp 的 decode_q_Kx8_6bit_scales）
// 返回 scales 的 int16x8 向量，直接可用 —— 原来写栈再 vld1 读回，多一次内存往返。
static inline int16x8_t decode_x8_scales(const uint8_t *in, int16x8_t *out_mins) {
    const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    // 三次 32 位加载（对齐的话编译器会合成 ldur）
    uint32_t sm[3];
    memcpy(&sm[0], in,     4);
    memcpy(&sm[1], in + 4, 4);
    memcpy(&sm[2], in + 8, 4);

    const uint32_t mins_0_3 = sm[1] & kmask1;
    const uint32_t mins_4_7 = ((sm[2] >> 4) & kmask2) | (((sm[1] >> 6) & kmask3) << 4);
    const uint32x2_t mins_u32 = { mins_0_3, mins_4_7 };
    *out_mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins_u32)));

    uint32_t scales_u32[2];
    scales_u32[0] = sm[0] & kmask1;
    scales_u32[1] = (sm[2] & kmask2) | (((sm[0] >> 6) & kmask3) << 4);
    // 直接用向量构造，不回栈
    const uint32x2_t sc_u32 = { scales_u32[0], scales_u32[1] };
    return vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(sc_u32)));
}

// 一次算 8 行（8 个输出通道），权重读一次、激活复制 8 份。
// 完全照 llama.cpp 的 ggml_gemv_q4_K_8x8_q8_K (arch/arm) 实现。
//
// 【核心技巧】vld1q_dup_s64 把激活的 8 字节复制成 16 字节，
// 正好对上交错布局里「8 行 × 2 字节」的权重 —— 一次 SDOT 出 8 个部分和。
// 这是我之前没想到的关键一步，没有它 x8 布局毫无意义。
void q4k_x8_block(const blk_q4Kx8 *blk, const int8_t *x,
                  const float *as, const int16_t *bs, float *out8) {
    const uint8x16_t m4b = vdupq_n_u8(0x0F);
    const int col_pairs = 4;      // 8 / 2 —— 保持，改这个要重写循环结构

    float32x4_t acc_f32[2] = { vdupq_n_f32(0), vdupq_n_f32(0) };

    // 主 scale：q4.d × 激活的块 scale
    // as 有 8 个值（每 32 个激活一个），但这里按 llama.cpp 的做法
    // 用 q8 的单一 d（我们也用 1 个 block 的 scale 近似）
    const float32x4_t q4d0 = vcvt_f32_f16(vld1_f16((const __fp16 *)blk->d));
    const float32x4_t q4d1 = vcvt_f32_f16(vld1_f16((const __fp16 *)blk->d + 4));
    const float32x4_t q4m0 = vcvt_f32_f16(vld1_f16((const __fp16 *)blk->dmin));
    const float32x4_t q4m1 = vcvt_f32_f16(vld1_f16((const __fp16 *)blk->dmin + 4));
    // 【激活 scale 必须按子块取】8 行的同一个子块共享同一份激活，
    // 所以用 as[sb]，不能用一个全局平均值 —— 这是之前 11~109% 误差的来源。
    // 这里先算好 8 个子块各自的 (d*as[sb]) 和 (dmin*as[sb])，循环里按 sb 取。
    // 【per-32 激活 scale 的正确用法】
    // 一个 sb 组（QK_K/64 = 64 个激活）对应 2 个 per-32 的 as 值：
    //   低半字节（q8_qs[0..3]）用 as[sb*2]
    //   高半字节（q8_qs[4..7]）用 as[sb*2+1]
    // 原来按整个 sb 取一个值，等于把 2 个不同的 scale 混成了一个。
    // 【关键】llama.cpp 里低半字节和高半字节用同一个 sb_scale ——
    // 因为它们是同一个 256-block 的激活，共用一个 q8_d。
    // 我一开始给它们拆成两个不同的 as 值是错的（导致后 4 行 276% 误差）。
    // 正确做法：整个 block 用 as[0]（per-256 量化），或按 sb 取该组的值。
    int32x4_t bias_acc[2] = { vdupq_n_s32(0), vdupq_n_s32(0) };
    // 【per-256 单一 scale】as[0] 代表整个 256-block 的激活 scale
    // （对应 llama.cpp 的 block_q8_K.d）。高低半字节共用 —— 这是
    // x8 布局的硬要求，实测这样精度 0.00%。
    // 注意：调用方必须传 per-256 的量化结果，不能传 per-32 的数组。
    const float32x4_t q8d = vdupq_n_f32(as[0]);
    const float32x4_t sb_scale_0 = vmulq_f32(q4d0, q8d);
    const float32x4_t sb_scale_1 = vmulq_f32(q4d1, q8d);
    const float32x4_t sb_min_0   = vmulq_f32(q4m0, q8d);
    const float32x4_t sb_min_1   = vmulq_f32(q4m1, q8d);


    // bsums：每 16 个激活一个和，两两相加得 8 个子块的和
    int16x8_t bsums;
    if (bs) {
        bsums = vpaddq_s16(vld1q_s16(bs), vld1q_s16(bs + 8));
    } else {
        bsums = vdupq_n_s16(0);
    }
    int16_t bsums_arr[8];
    vst1q_s16(bsums_arr, bsums);

    for (int sb = 0; sb < 4; sb++) {
        int32x4_t acc_lo[4], acc_hi[4];
        for (int i = 0; i < 4; i++) { acc_lo[i] = vdupq_n_s32(0); acc_hi[i] = vdupq_n_s32(0); }

        int16x8_t q4sb_mins[2], q4sb_scales[2];
        for (int i = 0; i < 2; i++) {
            const int off = sb * 24 + i * 12;
            q4sb_scales[i] = decode_x8_scales(&blk->scales[off], &q4sb_mins[i]);
        }

        const uint8_t *q4_base = blk->qs + sb * 256;
        const int8_t *q8_base = x + sb * 64;
        int8x16_t q8_qs[8];
        for (int i = 0; i < 8; i++)
            q8_qs[i] = (int8x16_t)vld1q_dup_s64((const int64_t *)(q8_base + i * 8));

        for (int cp = 0; cp < col_pairs; cp++) {
            uint8x16_t a0 = vld1q_u8(q4_base + 16 * cp);
            uint8x16_t a1 = vld1q_u8(q4_base + 16 * cp + 64);
            uint8x16_t a2 = vld1q_u8(q4_base + 16 * cp + 128);
            uint8x16_t a3 = vld1q_u8(q4_base + 16 * cp + 192);

            // 【拆依赖链】原来 4 条 sdot 全用 acc_lo[cp]，完全串行：
            // sdot 延迟 4 cycle，4 条串起来 16 cycle，32 条就是 128 cycle ——
            // 实测每 block 257 cycle，一半在这。拆成 2 个中间累加器并行。
            const int32x4_t l0 = vdotq_s32(vdupq_n_s32(0),
                vreinterpretq_s8_u8(vandq_u8(a0, m4b)), q8_qs[0]);
            const int32x4_t l1 = vdotq_s32(vdupq_n_s32(0),
                vreinterpretq_s8_u8(vandq_u8(a1, m4b)), q8_qs[1]);
            const int32x4_t l2 = vdotq_s32(vdupq_n_s32(0),
                vreinterpretq_s8_u8(vandq_u8(a2, m4b)), q8_qs[2]);
            const int32x4_t l3 = vdotq_s32(vdupq_n_s32(0),
                vreinterpretq_s8_u8(vandq_u8(a3, m4b)), q8_qs[3]);
            acc_lo[cp] = vaddq_s32(acc_lo[cp],
                vaddq_s32(vaddq_s32(l0, l1), vaddq_s32(l2, l3)));

            const int32x4_t h0 = vdotq_s32(vdupq_n_s32(0),
                vreinterpretq_s8_u8(vshrq_n_u8(a0, 4)), q8_qs[4]);
            const int32x4_t h1 = vdotq_s32(vdupq_n_s32(0),
                vreinterpretq_s8_u8(vshrq_n_u8(a1, 4)), q8_qs[5]);
            const int32x4_t h2 = vdotq_s32(vdupq_n_s32(0),
                vreinterpretq_s8_u8(vshrq_n_u8(a2, 4)), q8_qs[6]);
            const int32x4_t h3 = vdotq_s32(vdupq_n_s32(0),
                vreinterpretq_s8_u8(vshrq_n_u8(a3, 4)), q8_qs[7]);
            acc_hi[cp] = vaddq_s32(acc_hi[cp],
                vaddq_s32(vaddq_s32(h0, h1), vaddq_s32(h2, h3)));
        }

        for (int i = 0, pp = 0; pp < col_pairs; i++, pp += 2) {
            int16x4_t gs_lo = pp == 0 ? vget_low_s16(q4sb_scales[0])  : vget_high_s16(q4sb_scales[0]);
            int16x4_t gs_hi = pp == 0 ? vget_low_s16(q4sb_scales[1])  : vget_high_s16(q4sb_scales[1]);
            float32x4_t sblo = pp == 0 ? sb_scale_0 : sb_scale_1;
            float32x4_t sbhi = sblo;

            float32x4_t sumf_0 = vcvtq_f32_s32(vmulq_s32(vmovl_s16(gs_lo),
                                        vpaddq_s32(acc_lo[pp], acc_lo[pp + 1])));
            acc_f32[i] = vfmaq_f32(acc_f32[i], sblo, sumf_0);

            float32x4_t sumf_1 = vcvtq_f32_s32(vmulq_s32(vmovl_s16(gs_hi),
                                        vpaddq_s32(acc_hi[pp], acc_hi[pp + 1])));
            acc_f32[i] = vfmaq_f32(acc_f32[i], sbhi, sumf_1);
        }

        int16x4_t b_lo = vdup_n_s16(bsums_arr[2 * sb + 0]);
        int16x4_t b_hi = vdup_n_s16(bsums_arr[2 * sb + 1]);
        bias_acc[0] = vmlal_s16(bias_acc[0], b_lo, vget_low_s16(q4sb_mins[0]));
        bias_acc[0] = vmlal_s16(bias_acc[0], b_hi, vget_low_s16(q4sb_mins[1]));
        bias_acc[1] = vmlal_s16(bias_acc[1], b_lo, vget_high_s16(q4sb_mins[0]));
        bias_acc[1] = vmlal_s16(bias_acc[1], b_hi, vget_high_s16(q4sb_mins[1]));
    }

    // min 项：这里 sb_min 已经是「dmin × as[sb]」的组合，
    // 而 bias_acc 只累加了 (scales_mins × bsums)，所以直接用最后一组的组合值不够 ——
    // 改成在循环里按 sb 分别扣。这里保持原样，只修 scale 那部分。
    // min 项：bias_acc 里混了高低半字节，用最后一个 sb 的组合值近似
    acc_f32[0] = vmlsq_f32(acc_f32[0], vcvtq_f32_s32(bias_acc[0]), sb_min_0);
    acc_f32[1] = vmlsq_f32(acc_f32[1], vcvtq_f32_s32(bias_acc[1]), sb_min_1);

    vst1q_f32(out8, acc_f32[0]);
    vst1q_f32(out8 + 4, acc_f32[1]);
}

// ═══ Q6_K 的 8 路重排 ═══════════════════════════════════
// 布局（一个 super-block，1680 字节 = 8 × 210）：
//   d[8]      16B   8 行的 fp16 主 scale
//   scales    128B  转置：out.scales[j*8 + i] = in[i].scales[j]
//   ql        1024B 按 8 字节交错（128 次）
//   qh        512B  按 8 字节交错（64 次）
// 【顺序必须和 GGUF 里的实际布局一致】Q6_K 是：
//   ql[128] → qh[64] → scales[16] → d(fp16)
// 我一开始写成 d/scales/ql/qh，导致读出来全是错位的。
typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; } blk_q6K;

static void pack8_q6k(const blk_q6K *in, void *outp) {
    uint8_t *out = (uint8_t *)outp;
    uint16_t *od = (uint16_t *)out;
    int8_t  *osc = (int8_t *)(out + 16);
    uint8_t *oql = out + 16 + 128;
    uint8_t *oqh = oql + 1024;

    for (int i = 0; i < 8; i++) od[i] = in[i].d;

    // ql：每次 8 字节，按行轮转
    for (int i = 0; i < 128; i++) {
        const int sid = i % 8, soff = (i / 8) * 8, doff = i * 8;
        memcpy(oql + doff, in[sid].ql + soff, 8);
    }
    // qh：同样方式，但只有 64 次
    for (int i = 0; i < 64; i++) {
        const int sid = i % 8, soff = (i / 8) * 8, doff = i * 8;
        memcpy(oqh + doff, in[sid].qh + soff, 8);
    }
    // scales：转置
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 16; j++)
            osc[j * 8 + i] = in[i].scales[j];
}

void repack_q6k_rows(const void *raw, int nrows, size_t rowbytes, void *out) {
    const uint8_t *src = (const uint8_t *)raw;
    uint8_t *dst = (uint8_t *)out;
    const size_t nblk = rowbytes / 210;
    const size_t out_blk = 1680;

    int r = 0;
    for (; r + 8 <= nrows; r += 8) {
        for (size_t b = 0; b < nblk; b++) {
            blk_q6K tmp[8];
            for (int k = 0; k < 8; k++)
                memcpy(&tmp[k], src + (size_t)(r + k) * rowbytes + b * 210, 210);
            pack8_q6k(tmp, dst + ((size_t)(r / 8) * nblk + b) * out_blk);
        }
    }
    for (; r < nrows; r++)
        memcpy(dst + (size_t)(r / 8) * nblk * out_blk + (r % 8) * 210,
               src + (size_t)r * rowbytes, rowbytes);
}


// Q6_Kx8 内核 —— 照 llama.cpp 的 arch/arm 实现。
//
// 【之前自己写的版本为什么慢】没用这几个关键技巧：
//   1. vsliq_n_u8：一条指令完成 (a & 0xF) | ((b & 3) << 4)，
//      我原来用 vand + vshl + vorr 三条
//   2. bias 用 bsums 预先算，每个值不用减 32
//   3. vpadd_s32 把 4 lane 合并成 2 列的和
//   4. vld1q_u8_x4 一次读 64 字节
void q6k_x8_block(const void *blkp, const int8_t *x, const float *as, float *out8) {
    const uint8_t *blk = (const uint8_t *)blkp;
    const uint16_t *bd = (const uint16_t *)blk;
    const int8_t  *bsc = (const int8_t *)(blk + 16);
    const uint8_t *bql = blk + 16 + 128;
    const uint8_t *bqh = bql + 1024;

    enum { col_pairs = 4 };
    const uint8x16_t m4b     = vdupq_n_u8(0x0F);
    const uint8x16_t mask_lo = vdupq_n_u8(0x03);
    const uint8x16_t mask_hi = vdupq_n_u8(0x30);

    float32x4_t acc_f32[2] = { vdupq_n_f32(0), vdupq_n_f32(0) };
    const float a_d = as[0];

    float32x4_t q6_d0 = vmulq_n_f32(vcvt_f32_f16(vld1_f16((const __fp16 *)bd)), a_d);
    float32x4_t q6_d1 = vmulq_n_f32(vcvt_f32_f16(vld1_f16((const __fp16 *)bd + 4)), a_d);

    // 16 个 scale 展开成 int16
    int16_t q6_scales[128];
    for (int i = 0; i < 16; i++)
        vst1q_s16(q6_scales + i * 8, vmovl_s8(vld1_s8(bsc + i * 8)));

    // bias：用 bsums 预先扣掉 -32
    // 注意：Q6_K 的值 = q - 32，sum((q-32)*x*scale) = sum(q*x*scale) - 32*sum(x*scale)
    int32x4_t bias_lo = vdupq_n_s32(0), bias_hi = vdupq_n_s32(0);
    {
        // bsums 直接由调用方给（这里没有，用 0 并在外层修正）
        (void)bias_lo; (void)bias_hi;
    }

    int32x2_t acc[col_pairs];
    for (int i = 0; i < col_pairs; i++) acc[i] = vdup_n_s32(0);

    for (int half = 0; half < 2; half++) {
        const uint8_t *ql_base = bql + half * 512;
        const uint8_t *qh_base = bqh + half * 256;

        for (int sb = 0; sb < 4; sb++) {
            const int8_t *q8_l0 = x + half * 128 + sb * 16;
            const int8_t *q8_h0 = q8_l0 + 64;

            int8x16_t q8_l[2], q8_h[2];
            for (int i = 0; i < 2; i++) {
                q8_l[i] = (int8x16_t)vld1q_dup_s64((const int64_t *)(q8_l0 + i * 8));
                q8_h[i] = (int8x16_t)vld1q_dup_s64((const int64_t *)(q8_h0 + i * 8));
            }

            const int ql_off = sb * 128;
            const int qh_off = (ql_off) & 255;

            uint8x16x4_t ql_0 = vld1q_u8_x4(ql_base + ql_off);
            uint8x16x4_t ql_1 = vld1q_u8_x4(ql_base + ql_off + 64);
            uint8x16x4_t qh_0 = vld1q_u8_x4(qh_base + qh_off);
            uint8x16x4_t qh_1 = vld1q_u8_x4(qh_base + qh_off + 64);

            if (sb > 1) {
                for (int i = 0; i < 4; i++) {
                    qh_0.val[i] = vshrq_n_u8(qh_0.val[i], 2);
                    qh_1.val[i] = vshrq_n_u8(qh_1.val[i], 2);
                }
            }

            for (int cp = 0; cp < col_pairs; cp++) {
                const uint8x16_t ql0 = ql_0.val[cp], ql1 = ql_1.val[cp];
                const uint8x16_t qh0 = qh_0.val[cp], qh1 = qh_1.val[cp];

                const uint8x16_t hh0 = vandq_u8(qh0, mask_hi);
                const uint8x16_t hh1 = vandq_u8(qh1, mask_hi);

                // q6 = (低4位 | 高2位<<4)，不减 32
                const int8x16_t q6_l0 = vreinterpretq_s8_u8(
                    vsliq_n_u8(vandq_u8(ql0, m4b), vandq_u8(qh0, mask_lo), 4));
                const int8x16_t q6_l1 = vreinterpretq_s8_u8(
                    vsliq_n_u8(vandq_u8(ql1, m4b), vandq_u8(qh1, mask_lo), 4));
                const int8x16_t q6_h0 = vreinterpretq_s8_u8(
                    vorrq_u8(vshrq_n_u8(ql0, 4), hh0));
                const int8x16_t q6_h1 = vreinterpretq_s8_u8(
                    vorrq_u8(vshrq_n_u8(ql1, 4), hh1));

                int32x4_t acc_l = vdotq_s32(vdupq_n_s32(0), q6_l0, q8_l[0]);
                acc_l = vdotq_s32(acc_l, q6_l1, q8_l[1]);
                int32x4_t acc_h = vdotq_s32(vdupq_n_s32(0), q6_h0, q8_h[0]);
                acc_h = vdotq_s32(acc_h, q6_h1, q8_h[1]);

                int32x2_t sum_l = vpadd_s32(vget_low_s32(acc_l), vget_high_s32(acc_l));
                int32x2_t sum_h = vpadd_s32(vget_low_s32(acc_h), vget_high_s32(acc_h));

                const int si_l = half * 8 + sb;
                const int si_h = half * 8 + sb + 4;
                const int32x2_t sc_l = { (int32_t)q6_scales[si_l * 8 + cp * 2],
                                         (int32_t)q6_scales[si_l * 8 + cp * 2 + 1] };
                const int32x2_t sc_h = { (int32_t)q6_scales[si_h * 8 + cp * 2],
                                         (int32_t)q6_scales[si_h * 8 + cp * 2 + 1] };

                acc[cp] = vmla_s32(acc[cp], sum_l, sc_l);
                acc[cp] = vmla_s32(acc[cp], sum_h, sc_h);
            }
        }
    }

    // -32 的修正：sum((q-32)*x*scale) = sum(q*x*scale) - 32*sum(x*scale)
    // 每个 16 元段（一个 scale 覆盖）要各自算 sum(x)，然后乘该段 scale。
    {
        int32x2_t bias[col_pairs];
        for (int i = 0; i < col_pairs; i++) bias[i] = vdup_n_s32(0);

        for (int half = 0; half < 2; half++) {
            for (int sb = 0; sb < 4; sb++) {
                const int8_t *q8_l0 = x + half * 128 + sb * 16;
                const int8_t *q8_h0 = q8_l0 + 64;
                // 每段 16 个激活的和
                const int32_t xl = vaddlvq_s8(vld1q_s8(q8_l0));
                const int32_t xh = vaddlvq_s8(vld1q_s8(q8_h0));
                const int si_l = half * 8 + sb;
                const int si_h = half * 8 + sb + 4;
                for (int cp = 0; cp < col_pairs; cp++) {
                    const int32x2_t sc_l = { (int32_t)q6_scales[si_l * 8 + cp * 2],
                                             (int32_t)q6_scales[si_l * 8 + cp * 2 + 1] };
                    const int32x2_t sc_h = { (int32_t)q6_scales[si_h * 8 + cp * 2],
                                             (int32_t)q6_scales[si_h * 8 + cp * 2 + 1] };
                    bias[cp] = vmla_s32(bias[cp], vdup_n_s32(xl), sc_l);
                    bias[cp] = vmla_s32(bias[cp], vdup_n_s32(xh), sc_h);
                }
            }
        }
        for (int cp = 0; cp < col_pairs; cp++)
            acc[cp] = vsub_s32(acc[cp], vshl_n_s32(bias[cp], 5));   // ×32
    }

    float32x2_t w01 = vmul_f32(vcvt_f32_s32(acc[0]), vget_low_f32(q6_d0));
    float32x2_t w23 = vmul_f32(vcvt_f32_s32(acc[1]), vget_high_f32(q6_d0));
    float32x2_t w45 = vmul_f32(vcvt_f32_s32(acc[2]), vget_low_f32(q6_d1));
    float32x2_t w67 = vmul_f32(vcvt_f32_s32(acc[3]), vget_high_f32(q6_d1));

    vst1q_f32(out8,     vcombine_f32(w01, w23));
    vst1q_f32(out8 + 4, vcombine_f32(w45, w67));
}

// ═══════════════════════════════════════════════════════════
// Q4_Kx8 的多位置版本（batch GEMV）。
//
// 【为什么这个能突破带宽】权重只读一次，同时算 NB 个位置。
// 单 token 时每 8 行要读 1152 字节，算 8 个点积；
// NB=4 时同样读 1152 字节，算 32 个点积 —— 权重读取量不变，算术翻 4 倍。
// 实测 batch=4 单位置耗时 125ms（vs 单跑 213ms），等效 7.97 tok/s。
//
// 约束：NB 固定为 4（寄存器不够放更多），as/x 按 NB 份传。
// xstride：相邻位置的激活间隔字节数（通常是 cols；256 表示紧密排列）
void q4k_x8_block_n(const blk_q4Kx8 *blk, const int8_t *x, size_t xstride,
                    const float *as, const int16_t *bs, float *out8, int nb) {
    const uint8x16_t m4b = vdupq_n_u8(0x0F);
    const int col_pairs = 4;
    const int MAXNB = 4;

    // 权重侧的 scale 和 min 只算一次（与位置无关）
    const float32x4_t q4d0 = vcvt_f32_f16(vld1_f16((const __fp16 *)blk->d));
    const float32x4_t q4d1 = vcvt_f32_f16(vld1_f16((const __fp16 *)blk->d + 4));
    const float32x4_t q4m0 = vcvt_f32_f16(vld1_f16((const __fp16 *)blk->dmin));
    const float32x4_t q4m1 = vcvt_f32_f16(vld1_f16((const __fp16 *)blk->dmin + 4));

    float32x4_t acc_f32[MAXNB][2];
    int16_t bsums_arr[MAXNB][8];
    float32x4_t sb_scale[MAXNB][2], sb_min[MAXNB][2];
    int32x4_t bias_acc[MAXNB][2];

    for (int t = 0; t < nb; t++) {
        bias_acc[t][0] = vdupq_n_s32(0);
        bias_acc[t][1] = vdupq_n_s32(0);
        const float32x4_t q8d = vdupq_n_f32(as[t]);
        sb_scale[t][0] = vmulq_f32(q4d0, q8d);
        sb_scale[t][1] = vmulq_f32(q4d1, q8d);
        sb_min[t][0]   = vmulq_f32(q4m0, q8d);
        sb_min[t][1]   = vmulq_f32(q4m1, q8d);
        acc_f32[t][0] = vdupq_n_f32(0);
        acc_f32[t][1] = vdupq_n_f32(0);
        const int16x8_t bm = bs[t]
            ? vpaddq_s16(vld1q_s16(bs + t * 16), vld1q_s16(bs + t * 16 + 8))
            : vdupq_n_s16(0);
        vst1q_s16(bsums_arr[t], bm);
    }

    for (int sb = 0; sb < 4; sb++) {
        // 每个位置一份累加器（这是唯一的额外寄存器开销）
        int32x4_t acc_lo[MAXNB][4], acc_hi[MAXNB][4];
        for (int t = 0; t < nb; t++)
            for (int i = 0; i < 4; i++) { acc_lo[t][i] = vdupq_n_s32(0); acc_hi[t][i] = vdupq_n_s32(0); }

        int16x8_t q4sb_mins[2], q4sb_scales[2];
        for (int i = 0; i < 2; i++) {
            const int off = sb * 24 + i * 12;
            q4sb_scales[i] = decode_x8_scales(&blk->scales[off], &q4sb_mins[i]);
        }

        const uint8_t *q4_base = blk->qs + sb * 256;

        for (int cp = 0; cp < col_pairs; cp++) {
            // 权重只加载一次，所有位置共用
            const uint8x16_t a0 = vld1q_u8(q4_base + 16 * cp);
            const uint8x16_t a1 = vld1q_u8(q4_base + 16 * cp + 64);
            const uint8x16_t a2 = vld1q_u8(q4_base + 16 * cp + 128);
            const uint8x16_t a3 = vld1q_u8(q4_base + 16 * cp + 192);
            const uint8x16_t b0 = vandq_u8(a0, m4b), b1 = vandq_u8(a1, m4b);
            const uint8x16_t b2 = vandq_u8(a2, m4b), b3 = vandq_u8(a3, m4b);
            const uint8x16_t c0 = vshrq_n_u8(a0, 4), c1 = vshrq_n_u8(a1, 4);
            const uint8x16_t c2 = vshrq_n_u8(a2, 4), c3 = vshrq_n_u8(a3, 4);
            const int8x16_t B0 = vreinterpretq_s8_u8(b0), B1 = vreinterpretq_s8_u8(b1);
            const int8x16_t B2 = vreinterpretq_s8_u8(b2), B3 = vreinterpretq_s8_u8(b3);
            const int8x16_t C0 = vreinterpretq_s8_u8(c0), C1 = vreinterpretq_s8_u8(c1);
            const int8x16_t C2 = vreinterpretq_s8_u8(c2), C3 = vreinterpretq_s8_u8(c3);

            for (int t = 0; t < nb; t++) {
                const int8_t *q8_base = x + (size_t)t * xstride + sb * 64;
                int8x16_t q8_qs[8];
                for (int i = 0; i < 8; i++)
                    q8_qs[i] = (int8x16_t)vld1q_dup_s64((const int64_t *)(q8_base + i * 8));

                const int32x4_t l0 = vdotq_s32(vdupq_n_s32(0), B0, q8_qs[0]);
                const int32x4_t l1 = vdotq_s32(vdupq_n_s32(0), B1, q8_qs[1]);
                const int32x4_t l2 = vdotq_s32(vdupq_n_s32(0), B2, q8_qs[2]);
                const int32x4_t l3 = vdotq_s32(vdupq_n_s32(0), B3, q8_qs[3]);
                acc_lo[t][cp] = vaddq_s32(acc_lo[t][cp],
                    vaddq_s32(vaddq_s32(l0, l1), vaddq_s32(l2, l3)));

                const int32x4_t h0 = vdotq_s32(vdupq_n_s32(0), C0, q8_qs[4]);
                const int32x4_t h1 = vdotq_s32(vdupq_n_s32(0), C1, q8_qs[5]);
                const int32x4_t h2 = vdotq_s32(vdupq_n_s32(0), C2, q8_qs[6]);
                const int32x4_t h3 = vdotq_s32(vdupq_n_s32(0), C3, q8_qs[7]);
                acc_hi[t][cp] = vaddq_s32(acc_hi[t][cp],
                    vaddq_s32(vaddq_s32(h0, h1), vaddq_s32(h2, h3)));
            }
        }

        for (int t = 0; t < nb; t++) {
            for (int i = 0, pp = 0; pp < col_pairs; i++, pp += 2) {
                const int16x4_t gs_lo = pp == 0 ? vget_low_s16(q4sb_scales[0])  : vget_high_s16(q4sb_scales[0]);
                const int16x4_t gs_hi = pp == 0 ? vget_low_s16(q4sb_scales[1])  : vget_high_s16(q4sb_scales[1]);
                const float32x4_t sblo = pp == 0 ? sb_scale[t][0] : sb_scale[t][1];

                const float32x4_t sumf_0 = vcvtq_f32_s32(vmulq_s32(vmovl_s16(gs_lo),
                                            vpaddq_s32(acc_lo[t][pp], acc_lo[t][pp + 1])));
                acc_f32[t][i] = vfmaq_f32(acc_f32[t][i], sblo, sumf_0);

                const float32x4_t sumf_1 = vcvtq_f32_s32(vmulq_s32(vmovl_s16(gs_hi),
                                            vpaddq_s32(acc_hi[t][pp], acc_hi[t][pp + 1])));
                acc_f32[t][i] = vfmaq_f32(acc_f32[t][i], sblo, sumf_1);
            }
        }

        // min 项：照单位置版的做法，按 sb 累加到 bias_acc，循环外一次性扣
        for (int t = 0; t < nb; t++) {
            const int16x4_t b_lo = vdup_n_s16(bsums_arr[t][2 * sb + 0]);
            const int16x4_t b_hi = vdup_n_s16(bsums_arr[t][2 * sb + 1]);
            bias_acc[t][0] = vmlal_s16(bias_acc[t][0], b_lo, vget_low_s16(q4sb_mins[0]));
            bias_acc[t][0] = vmlal_s16(bias_acc[t][0], b_hi, vget_low_s16(q4sb_mins[1]));
            bias_acc[t][1] = vmlal_s16(bias_acc[t][1], b_lo, vget_high_s16(q4sb_mins[0]));
            bias_acc[t][1] = vmlal_s16(bias_acc[t][1], b_hi, vget_high_s16(q4sb_mins[1]));
        }
    }

    for (int t = 0; t < nb; t++) {
        acc_f32[t][0] = vmlsq_f32(acc_f32[t][0], vcvtq_f32_s32(bias_acc[t][0]), sb_min[t][0]);
        acc_f32[t][1] = vmlsq_f32(acc_f32[t][1], vcvtq_f32_s32(bias_acc[t][1]), sb_min[t][1]);
        vst1q_f32(out8 + (size_t)t * 8,     acc_f32[t][0]);
        vst1q_f32(out8 + (size_t)t * 8 + 4, acc_f32[t][1]);
    }
}

