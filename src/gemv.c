// 融合"解量化 + 点积"的 GEMV：decode 阶段唯一的性能热点。
//
// 【为什么要融合，不能分两步】
// 分两步 = 先把 2.5GB 权重解成 float（10GB！）再算，内存直接爆。
// 就算解一小块到临时 buffer，也是「读权重 → 写 buffer → 读 buffer → 计算」，
// 多一次写和一次读，而 decode 本来就是内存受限，这会让带宽白白浪费一倍。
// 融合后在寄存器里解完立刻 vdot，权重只从内存读一次。
//
// 【Q4_K 的点积要处理 min（偏移）】
// Q4_K 是「非对称」量化：值 = d*sc*q - dmin*m。
// 所以 sum(w*x) = sum((d*sc*q - dmin*m) * x)
//              = sum(d*sc*q*x) - sum(dmin*m*x)
// 第一项能用 int 累加（q 是 int4，x 是 int8），
// 第二项要把 x 按块求和再乘常数 —— 所以每算完一个子块要单独处理 min。
// 这里选择「解到 int8 再 SDOT」而不是纯整数方案：
// 纯整数方案要把 d、dmin 都拆成整数乘子，精度损失和代码复杂度都不划算。
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <arm_neon.h>
#include <stdalign.h>

// fp16 -> float：64KB 查表，避免每块 2 次标量位运算。
// 一块 144 字节里要转两次，2.5GB 模型共 1700 万块 —— 标量版本这里就是大头。
static float fp16_tbl[65536];
static int   fp16_tbl_ready = 0;
static void init_fp16_tbl(void) {
    // 用编译器内建的 fp16 转换 —— 手写位运算在非规格化数上容易错
    for (int i = 0; i < 65536; i++) {
        __fp16 h;
        memcpy(&h, &i, 2);
        fp16_tbl[i] = (float)h;
    }
    fp16_tbl_ready = 1;
}
static inline float half2f_fast(uint16_t h) {
    float16_t hv;
    memcpy(&hv, &h, 2);
    return (float)hv;
}

static inline void get_scale_min_k4(int j, const uint8_t *q,
                                    uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Q4_K 权重（n 个）· int8 激活 → float
//
// 【为什么这一版比"解到栈上再点积"快得多】
// 前一版每 32 个权重都往栈上的 tmp[32] 写一遍再读回来，
// 光是这次写+读就把内存带宽吃掉一大半（实测只有 0.88 GB/s）。
// 正确做法：全程在 NEON 寄存器里把 4-bit 摊成 int8，直接 SDOT。
//
// 【一个 64 权重的循环体】qs 读 2 次（g=0,1），每次同时服务低/高半字节，
// 激活也只读一次 —— 不重复加载。

// 激活按 16 个一组求和，供 Q4_K 的 min 项使用。
//
// 【为什么必须预计算】
// Q4_K 的 min 项要 sum(x) 按子块（32 个）加权，
// 原来在热循环里每块现算 4 次 vaddw_s8 —— 但 x 只跟当前 token 有关，
// 跟权重无关。同一层的 5 个 GEMV、上一层算出的激活，全都共用这一份。
// 每个 token 会把整个模型过一遍、几万个块各算一次，白白重复几万次。
// llama.cpp 就是靠 block_q8_K 里预存的 bsums 省掉这笔开销。
//
// out 长度需要 ceil(n/16)，每个元素 = 该 16 个 int8 的和。
void act_bsums(const int8_t *x, int16_t *out, size_t n) {
    size_t i = 0, o = 0;
    for (; i + 16 <= n; i += 16, o++) {
        // 直接累加 —— 只有 16 项，编译器会自动向量化，
        // 手写 vpadd 链条反而容易搞错配对顺序（踩过）。
        const int8x16_t v = vld1q_s8(x + i);
        out[o] = (int16_t)vaddlvq_s8(v);
    }
    for (; i < n; i++, o++) out[o] = x[i];
}

float q4k_dot(const void *raw, const int8_t *x, size_t n) {
    const uint8_t *p = (const uint8_t *)raw;
    size_t nblk = n / 256;
    // 4 个独立累加器 —— 单个 float 累加器会形成 FADD 依赖链（A77 上 4 cycle），
    // 每 256 权重只加 8 次，8×4=32 cycle 远超 144 字节的解码时间，
    // 会把整个循环卡在等浮点上。拆成 4 条链后能并行发射。
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

    const uint8x16_t mask4 = vdupq_n_u8(0x0F);

    for (size_t b = 0; b < nblk; b++, p += 144) {
        uint16_t dh, dmh;
        memcpy(&dh, p, 2);
        memcpy(&dmh, p + 2, 2);
        const float d = half2f_fast(dh);
        const float dmin = half2f_fast(dmh);
        const uint8_t *scales = p + 4;
        const uint8_t *qs = p + 16;
        const int8_t *xp = x + b * 256;

        float ds[8], dm[8];
        for (int i = 0; i < 8; i++) {
            uint8_t sc, mn;
            get_scale_min_k4(i, scales, &sc, &mn);
            ds[i] = d * sc;
            dm[i] = dmin * mn;
        }

        for (int j = 0; j < 256; j += 64) {
            const int si = j / 32;

            int32x4_t accLo = vdupq_n_s32(0), accHi = vdupq_n_s32(0);

            for (int g = 0; g < 2; g++) {
                const uint8x16_t packed = vld1q_u8(qs + g * 16);
                accLo = vdotq_s32(accLo, vreinterpretq_s8_u8(vandq_u8(packed, mask4)),
                                  vld1q_s8(xp + j + g * 16));
                accHi = vdotq_s32(accHi, vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)),
                                  vld1q_s8(xp + j + 32 + g * 16));
            }

            int16x8_t xsumLo = vdupq_n_s16(0), xsumHi = vdupq_n_s16(0);
            for (int g = 0; g < 2; g++) {
                const int8x16_t xlo = vld1q_s8(xp + j + g * 16);
                const int8x16_t xhi = vld1q_s8(xp + j + 32 + g * 16);
                xsumLo = vaddw_s8(xsumLo, vget_low_s8(xlo));
                xsumLo = vaddw_high_s8(xsumLo, xlo);
                xsumHi = vaddw_s8(xsumHi, vget_low_s8(xhi));
                xsumHi = vaddw_high_s8(xsumHi, xhi);
            }
            const float v0 = (float)vaddvq_s32(accLo) * ds[si + 0] - (float)vaddvq_s16(xsumLo) * dm[si + 0];
            const float v1 = (float)vaddvq_s32(accHi) * ds[si + 1] - (float)vaddvq_s16(xsumHi) * dm[si + 1];
            switch ((j / 64) & 3) {
            case 0: acc0 += v0; acc1 += v1; break;
            case 1: acc2 += v0; acc3 += v1; break;
            case 2: acc0 += v0; acc2 += v1; break;
            default: acc1 += v0; acc3 += v1; break;
            }

            qs += 32;
        }
    }
    return (acc0 + acc1) + (acc2 + acc3);
}


// 带预计算 bsums 的版本 —— 引擎实际用它。
// bs 长度 >= n/16，由 act_bsums 一次算好，跨所有 GEMV 复用。
float q4k_dot_bs(const void *raw, const int8_t *x, const int16_t *bs, size_t n) {
    const uint8_t *p = (const uint8_t *)raw;
    size_t nblk = n / 256;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

    const uint8x16_t mask4 = vdupq_n_u8(0x0F);

    for (size_t b = 0; b < nblk; b++, p += 144) {
        uint16_t dh, dmh;
        memcpy(&dh, p, 2);
        memcpy(&dmh, p + 2, 2);
        const float d = half2f_fast(dh);
        const float dmin = half2f_fast(dmh);
        const uint8_t *scales = p + 4;
        const uint8_t *qs = p + 16;
        const int8_t *xp = x + b * 256;

        float ds[8], dm[8];
        for (int i = 0; i < 8; i++) {
            uint8_t sc, mn;
            get_scale_min_k4(i, scales, &sc, &mn);
            ds[i] = d * sc;
            dm[i] = dmin * mn;
        }

        // 这个 super-block 的 16 组 bsums（每 16 个激活一个和）
        const int16_t *bp = bs + b * 16;

        for (int j = 0; j < 256; j += 64) {
            const int si = j / 32;
            int32x4_t accLo = vdupq_n_s32(0), accHi = vdupq_n_s32(0);

            for (int g = 0; g < 2; g++) {
                const uint8x16_t packed = vld1q_u8(qs + g * 16);
                const int8x16_t xlo = vld1q_s8(xp + j + g * 16);
                const int8x16_t xhi = vld1q_s8(xp + j + 32 + g * 16);
                accLo = vdotq_s32(accLo, vreinterpretq_s8_u8(vandq_u8(packed, mask4)), xlo);
                accHi = vdotq_s32(accHi, vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)), xhi);
            }

            // 子块 si 覆盖激活 [si*32, si*32+32)，即 bsums 的 [si*2, si*2+2)
            const int32_t sumLo = bp[(si + 0) * 2] + bp[(si + 0) * 2 + 1];
            const int32_t sumHi = bp[(si + 1) * 2] + bp[(si + 1) * 2 + 1];

            const float v0 = (float)vaddvq_s32(accLo) * ds[si + 0] - (float)sumLo * dm[si + 0];
            const float v1 = (float)vaddvq_s32(accHi) * ds[si + 1] - (float)sumHi * dm[si + 1];
            switch ((j / 64) & 3) {
            case 0: acc0 += v0; acc1 += v1; break;
            case 1: acc2 += v0; acc3 += v1; break;
            case 2: acc0 += v0; acc2 += v1; break;
            default: acc1 += v0; acc3 += v1; break;
            }

            qs += 32;
        }
    }
    return (acc0 + acc1) + (acc2 + acc3);
}

// Q6_K 权重（n 个）· int8 激活 → float
//
// Q6_K 是对称量化（值 = d*sc*q，q 在 -32..31），没有 min 项。
//
// 【官方布局的确切含义】对每个 half（128 个值）：
//   for l in 0..31:
//     q1 = (ql[l+0]&0xF) | ((qh[l]>>0 & 3)<<4)   → 位置 l+0   , sc[is+0]
//     q2 = (ql[l+32]&0xF) | ((qh[l]>>2 & 3)<<4)  → 位置 l+32  , sc[is+2]
//     q3 = (ql[l+0]>>4)   | ((qh[l]>>4 & 3)<<4)  → 位置 l+64  , sc[is+4]
//     q4 = (ql[l+32]>>4)  | ((qh[l]>>6 & 3)<<4)  → 位置 l+96  , sc[is+6]
//   其中 is = l/16（所以 scale 每 16 个一换，l=0..15 和 l=16..31 各一档）
//
// 【一个 l 组产出 4 段】每段 32 长但都由同一批 ql[l] 驱动，
// 所以按 l 的 16 个一组处理，一次生成 4 个 int8x16 直接 SDOT。

// ── Q2_K 融合 GEMV ──────────────────────────────────────
// 2-bit 值域 0..3，没有符号。结构：256 权重/84 字节
//   scales[16]: 每字节装 (scale & 0xF) | (min << 4)，16 组各管 16 个权重
//   qs[64]:     每字节 4 个 2-bit，按 shift=0/2/4/6 取
// 解出来是 d*(sc&0xF)*q - dmin*(sc>>4)，和 Q4_K 一样是"非对称"。
float q2k_dot(const void *raw, const int8_t *x, size_t n) {
    const uint8_t *p = (const uint8_t *)raw;
    size_t nblk = n / 256;
    float acc0 = 0.f, acc1 = 0.f;

    for (size_t b = 0; b < nblk; b++, p += 84) {
        const uint8_t *sc = p;
        const uint8_t *qs = p + 16;
        uint16_t dh, dmh;
        memcpy(&dh, p + 80, 2);
        memcpy(&dmh, p + 82, 2);
        const float d = half2f_fast(dh);
        const float dmin = half2f_fast(dmh);
        const int8_t *xp = x + b * 256;

        int is = 0, w = 0;
        for (int half = 0; half < 2; half++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                // 前 16 个
                const uint8_t s0 = sc[is++];
                int32x4_t a0 = vdupq_n_s32(0);
                int16x8_t x0 = vdupq_n_s16(0);
                {
                    alignas(16) int8_t t[16];
                    const uint8x16_t v = vld1q_u8(qs);
                    const uint8x16_t q2 = vandq_u8(vshlq_u8(v, vdupq_n_s8((int8_t)(-shift))), vdupq_n_u8(3));
                    vst1q_s8(t, vreinterpretq_s8_u8(q2));
                    a0 = vdotq_s32(a0, vld1q_s8(t), vld1q_s8(xp + w));
                    x0 = vaddw_s8(x0, vget_low_s8(vld1q_s8(xp + w)));
                    x0 = vaddw_high_s8(x0, vld1q_s8(xp + w));
                }
                acc0 += (float)vaddvq_s32(a0) * (d * (s0 & 0xF))
                      - (float)vaddvq_s16(x0) * (dmin * (s0 >> 4));

                // 后 16 个
                const uint8_t s1 = sc[is++];
                int32x4_t a1 = vdupq_n_s32(0);
                int16x8_t x1 = vdupq_n_s16(0);
                {
                    alignas(16) int8_t t[16];
                    const uint8x16_t v = vld1q_u8(qs + 16);
                    const uint8x16_t q2 = vandq_u8(vshlq_u8(v, vdupq_n_s8((int8_t)(-shift))), vdupq_n_u8(3));
                    vst1q_s8(t, vreinterpretq_s8_u8(q2));
                    a1 = vdotq_s32(a1, vld1q_s8(t), vld1q_s8(xp + w + 16));
                    x1 = vaddw_s8(x1, vget_low_s8(vld1q_s8(xp + w + 16)));
                    x1 = vaddw_high_s8(x1, vld1q_s8(xp + w + 16));
                }
                acc1 += (float)vaddvq_s32(a1) * (d * (s1 & 0xF))
                      - (float)vaddvq_s16(x1) * (dmin * (s1 >> 4));

                shift += 2;
                w += 32;
            }
            qs += 32;
        }
    }
    return acc0 + acc1;
}

// ── Q3_K 融合 GEMV ──────────────────────────────────────
// 结构：256 权重/110 字节。hmask 提供第 3 位（有则 0，无则减 4）
float q3k_dot(const void *raw, const int8_t *x, size_t n) {
    const uint8_t *p = (const uint8_t *)raw;
    size_t nblk = n / 256;
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    float acc = 0.f;

    for (size_t b = 0; b < nblk; b++, p += 110) {
        const uint8_t *hm = p;
        const uint8_t *qs = p + 32;
        uint16_t dh;
        memcpy(&dh, p + 108, 2);
        const float d_all = half2f_fast(dh);

        uint32_t aux[4];
        memcpy(aux, p + 96, 12);
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t *scales = (const int8_t *)aux;

        const int8_t *xp = x + b * 256;
        uint8_t m = 1;
        int is = 0, w = 0;
        for (int half = 0; half < 2; half++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                for (int seg = 0; seg < 2; seg++) {
                    alignas(16) int8_t t[16];
                    const uint8x16_t v = vld1q_u8(qs + seg * 16);
                    const uint8x16_t lo = vandq_u8(vshlq_u8(v, vdupq_n_s8((int8_t)(-shift))), vdupq_n_u8(3));
                    const uint8x16_t hi = vld1q_u8(hm + seg * 16);
                    // (hm & m) ? 0 : -4
                    const uint8x16_t notset = vtstq_u8(hi, vdupq_n_u8(m));
                    int8x16_t val = vreinterpretq_s8_u8(lo);
                    val = vsubq_s8(val, vandq_s8(vmvnq_u8(notset), vdupq_n_s8(4)));
                    vst1q_s8(t, val);

                    int32x4_t a = vdotq_s32(vdupq_n_s32(0), vld1q_s8(t), vld1q_s8(xp + w));
                    acc += (float)vaddvq_s32(a) * (d_all * (scales[is] - 32));

                    is++;
                    w += 16;
                }
                shift += 2;
                m <<= 1;
            }
            qs += 32;
        }
    }
    return acc;
}

float q6k_dot(const void *raw, const int8_t *x, size_t n) {
    const uint8_t *p = (const uint8_t *)raw;
    size_t nblk = n / 256;
    float total = 0.0f;

    const uint8x16_t m0F = vdupq_n_u8(0x0F);
    const uint8x16_t m03 = vdupq_n_u8(0x03);
    const int8x16_t k32 = vdupq_n_s8(32);

    for (size_t b = 0; b < nblk; b++, p += 210) {
        uint16_t dh;
        memcpy(&dh, p + 208, 2);
        const float d = half2f_fast(dh);
        const uint8_t *ql = p;
        const uint8_t *qh = p + 128;
        const int8_t *sc = (const int8_t *)(p + 192);
        const int8_t *xp = x + b * 256;

        for (int half = 0; half < 2; half++) {
            const int8_t *xb = xp + half * 128;

            for (int h = 0; h < 2; h++) {      // l = 0..15 和 16..31
                const int l0 = h * 16;
                // ql 这边：前 16 字节管 q1/q3（位置 l+0 / l+64），
                //           后 16 字节（ql+32）管 q2/q4（位置 l+32 / l+96）
                const uint8x16_t qla = vld1q_u8(ql + l0);
                const uint8x16_t qlb = vld1q_u8(ql + 32 + l0);
                // qh 的 16 字节就对应这 16 个 l
                const uint8x16_t qhv = vld1q_u8(qh + l0);

                // 【少一步移位】h0 不用移位，h3 不用 and（右移 6 位后高 2 位清零）。
                // 原来 h0/h1/h2 各一次 vand + h1/h2 各一次 vshr = 5 条，
                // 现在 h0 直接 vand、h3 直接 vshr，少一次 and。
                const uint8x16_t h0 = vandq_u8(qhv, m03);
                const uint8x16_t h1 = vandq_u8(vshrq_n_u8(qhv, 2), m03);
                const uint8x16_t h2 = vandq_u8(vshrq_n_u8(qhv, 4), m03);
                const uint8x16_t h3 = vshrq_n_u8(qhv, 6);

                // 用 vbsl 拼位（一次搞定 or）：vbsl(mask, a, b) = (mask & a) | (~mask & b)
                // 这里要高 4 位 = ql 低 4 位，低 4 位 = qh << 4
                // 其实 vaddq 和 vorr 等价（高低位不重叠），保持 vaddq 更快
                int8x16_t v0 = vaddq_s8(vreinterpretq_s8_u8(vandq_u8(qla, m0F)),
                                        vreinterpretq_s8_u8(vshlq_n_u8(h0, 4)));
                int8x16_t v1 = vaddq_s8(vreinterpretq_s8_u8(vandq_u8(qlb, m0F)),
                                        vreinterpretq_s8_u8(vshlq_n_u8(h1, 4)));
                int8x16_t v2 = vaddq_s8(vreinterpretq_s8_u8(vshrq_n_u8(qla, 4)),
                                        vreinterpretq_s8_u8(vshlq_n_u8(h2, 4)));
                int8x16_t v3 = vaddq_s8(vreinterpretq_s8_u8(vshrq_n_u8(qlb, 4)),
                                        vreinterpretq_s8_u8(vshlq_n_u8(h3, 4)));

                v0 = vsubq_s8(v0, k32);
                v1 = vsubq_s8(v1, k32);
                v2 = vsubq_s8(v2, k32);
                v3 = vsubq_s8(v3, k32);

                // 激活：4 段各在 xb + {0,32,64,96} + l0
                // 【先发 4 个 vdot，再统一横向求和】
                // 原来每算一个就 vaddvq_s32，等于强制串行 —— vaddvq 有 5~7 cycle 延迟，
                // 四次串起来就是 20+ cycle 的停顿。分开写让 CPU 能交错发射。
                const int32x4_t t0 = vdotq_s32(vdupq_n_s32(0), v0, vld1q_s8(xb + 0 + l0));
                const int32x4_t t1 = vdotq_s32(vdupq_n_s32(0), v1, vld1q_s8(xb + 32 + l0));
                const int32x4_t t2 = vdotq_s32(vdupq_n_s32(0), v2, vld1q_s8(xb + 64 + l0));
                const int32x4_t t3 = vdotq_s32(vdupq_n_s32(0), v3, vld1q_s8(xb + 96 + l0));
                const int32_t s0 = vaddvq_s32(t0);
                const int32_t s1 = vaddvq_s32(t1);
                const int32_t s2 = vaddvq_s32(t2);
                const int32_t s3 = vaddvq_s32(t3);

                // scale：is = l/16 = h，所以四段各取 sc[half*8 + h + {0,2,4,6}]
                total += (float)s0 * d * sc[h + 0];
                total += (float)s1 * d * sc[h + 2];
                total += (float)s2 * d * sc[h + 4];
                total += (float)s3 * d * sc[h + 6];
            }

            ql += 64; qh += 32; sc += 8;
        }
    }
    return total;
}

// ── 批量 GEMV → GEMM ────────────────────────────────────
// 【为什么要这个】投机解码验证时，一次前向要同时算 k 个位置。
// 如果按 token 循环调 gemv，权重会被解 k 遍 —— 而 decode 本来就卡在
// 解量化的 ALU 上，k 倍开销直接把投机解码的收益吃光。
// 这里改成：每行权重解一次成 int8，再和 batch 内所有激活各做一次 SDOT。
// 权重读 1 次、解 1 次，激活（很小，在 L1）读 k 次。

// Q4_K 批量：W[rows x cols] · X[n x cols] -> Y[n x rows]
// bs 是每个 token 的 bsums，布局 [n][cols/16]
// asc 非空时按 per-32 激活 scale 算（和单 token 路径的 q4k_dot_bsb 一致），
// 每个 256-block 用 asc[8] 这 8 个值；asc 为空则退回 per-tensor（结果外部乘 scale）。
void q4k_gemm_row(const void *raw, const int8_t *X, const int16_t *bs,
                  float *Y, int n, size_t cols, size_t ldY);
void q4k_gemm_row_a(const void *raw, const int8_t *X, const int16_t *bs,
                    const float *asc, float *Y, int n, size_t cols, size_t ldY) {
    const uint8_t *p = (const uint8_t *)raw;
    const size_t nblk = cols / 256;
    const uint8x16_t mask4 = vdupq_n_u8(0x0F);

    for (int t = 0; t < n; t++) Y[(size_t)t * ldY] = 0.f;

    for (size_t b = 0; b < nblk; b++, p += 144) {
        uint16_t dh, dmh;
        memcpy(&dh, p, 2);
        memcpy(&dmh, p + 2, 2);
        const float d = half2f_fast(dh);
        const float dmin = half2f_fast(dmh);
        const uint8_t *scales = p + 4;
        const uint8_t *qs = p + 16;

        float ds[8], dm[8];
        for (int i = 0; i < 8; i++) {
            uint8_t sc, mn;
            get_scale_min_k4(i, scales, &sc, &mn);
            ds[i] = d * sc;
            dm[i] = dmin * mn;
        }

        // 权重解一次，8 个子块各 32 个 int8。
        // 【布局】一个 64 权重的循环体：qs 的 32 字节里，
        //   qs[0..15] 的低半字节  -> 子块 si+0 的前 16 个
        //   qs[0..15] 的高半字节  -> 子块 si+1 的前 16 个
        //   qs[16..31] 的低半字节 -> 子块 si+0 的后 16 个
        //   qs[16..31] 的高半字节 -> 子块 si+1 的后 16 个
        // 所以 g 轮（g=0 是前 16、g=1 是后 16）分别填 w[si] 的两半。
        // 严格照搬单 token 版的结构：
        //   j 走 0,64,128,192（每轮 64 个权重）
        //   si = j/32，qs 游标每轮前进 32 字节
        //   段内 g=0 用 qs[0..15] 的低半字节 → 子块 si+0 的 [j+g*16, +16)
        //            高半字节 → 子块 si+1 的 [j+32+g*16, +16)
        alignas(16) int8_t w[8][32];
        for (int j = 0; j < 256; j += 64) {
            const int si = j / 32;
            const uint8_t *q = qs + (j / 64) * 32;
            for (int g = 0; g < 2; g++) {
                const uint8x16_t a = vld1q_u8(q + g * 16);
                vst1q_s8(w[si]     + g * 16, vreinterpretq_s8_u8(vandq_u8(a, mask4)));
                vst1q_s8(w[si + 1] + g * 16, vreinterpretq_s8_u8(vshrq_n_u8(a, 4)));
            }
        }

        for (int t = 0; t < n; t++) {
            const int8_t *xp = X + (size_t)t * cols + b * 256;
            const int16_t *bp = bs ? bs + (size_t)t * (cols / 16 + 1) + b * 16 : NULL;
            const float *ap = asc ? asc + (size_t)t * (cols / 32 + 1) + b * 8 : NULL;
            float acc = Y[(size_t)t * ldY];

            for (int si = 0; si < 8; si++) {
                // 子块是 32 个权重，vdotq 一次 16 个，所以要跑两趟
                const int8_t *ws = w[si];
                const int8_t *xs = xp + si * 32;
                int32x4_t a = vdotq_s32(vdupq_n_s32(0), vld1q_s8(ws),      vld1q_s8(xs));
                a = vdotq_s32(a, vld1q_s8(ws + 16), vld1q_s8(xs + 16));
                const int32_t dt = vaddvq_s32(a);

                int32_t ssum;
                if (bp) {
                    ssum = bp[si * 2] + bp[si * 2 + 1];
                } else {
                    ssum = 0;
                    for (int k = 0; k < 32; k++) ssum += xs[k];
                }
                const float am = ap ? ap[si] : 1.0f;
                acc += (float)dt * ds[si] * am - (float)ssum * dm[si] * am;
            }
            Y[(size_t)t * ldY] = acc;
        }
        // 一个 block 的 qs 是 128 字节；下一轮 p 前进 144，
        // 所以这里把 qs 也推到同一位置，避免用旧指针
        qs = p + 16 + 144;
    }
}

void q4k_gemm_row(const void *raw, const int8_t *X, const int16_t *bs,
                  float *Y, int n, size_t cols, size_t ldY) {
    q4k_gemm_row_a(raw, X, bs, NULL, Y, n, cols, ldY);
}

// Q6_K 批量（对称量化，没有 min 项，比 Q4_K 简单）
// asc 非空时按 per-32 激活 scale（每 256 个 block 有 8 个），结果已是最终值。
// Q6_K 的权重 scale 每 16 个一组，而激活 scale 每 32 个一组 —— 一个 32 组
// 正好覆盖两个 16 组，所以同一个 am 用两次。
void q6k_gemm_row_a(const void *raw, const int8_t *X, const float *asc, float *Y,
                    int n, size_t cols, size_t ldY) {
    const uint8_t *p = (const uint8_t *)raw;
    const size_t nblk = cols / 256;
    const uint8x16_t m0F = vdupq_n_u8(0x0F);
    const uint8x16_t m03 = vdupq_n_u8(0x03);
    const int8x16_t  k32 = vdupq_n_s8(32);

    for (int t = 0; t < n; t++) Y[(size_t)t * ldY] = 0.f;

    for (size_t b = 0; b < nblk; b++, p += 210) {
        uint16_t dh;
        memcpy(&dh, p + 208, 2);
        const float d = half2f_fast(dh);
        const uint8_t *ql = p;
        const uint8_t *qh = p + 128;
        const int8_t *sc = (const int8_t *)(p + 192);

        // 解出一整个 super-block 的 256 个 int8。
        // 严格照单 token 版的落点：half → [half*128, +128)，
        // 组内 l 走 0..31 产出四个位置 l+0 / l+32 / l+64 / l+96。
        alignas(16) int8_t w[256];
        {
            const uint8_t *qlp = ql, *qhp = qh;
            for (int half = 0; half < 2; half++) {
                int8_t *dst = w + half * 128;
                for (int h = 0; h < 2; h++) {
                    const int l0 = h * 16;
                    const uint8x16_t qla = vld1q_u8(qlp + l0);
                    const uint8x16_t qlb = vld1q_u8(qlp + 32 + l0);
                    const uint8x16_t qhv = vld1q_u8(qhp + l0);
                    const uint8x16_t h0 = vandq_u8(qhv, m03);
                    const uint8x16_t h1 = vandq_u8(vshrq_n_u8(qhv, 2), m03);
                    const uint8x16_t h2 = vandq_u8(vshrq_n_u8(qhv, 4), m03);
                    const uint8x16_t h3 = vshrq_n_u8(qhv, 6);
                    int8x16_t v0 = vsubq_s8(vaddq_s8(vreinterpretq_s8_u8(vandq_u8(qla, m0F)),
                                                    vreinterpretq_s8_u8(vshlq_n_u8(h0, 4))), k32);
                    int8x16_t v1 = vsubq_s8(vaddq_s8(vreinterpretq_s8_u8(vandq_u8(qlb, m0F)),
                                                    vreinterpretq_s8_u8(vshlq_n_u8(h1, 4))), k32);
                    int8x16_t v2 = vsubq_s8(vaddq_s8(vreinterpretq_s8_u8(vshrq_n_u8(qla, 4)),
                                                    vreinterpretq_s8_u8(vshlq_n_u8(h2, 4))), k32);
                    int8x16_t v3 = vsubq_s8(vaddq_s8(vreinterpretq_s8_u8(vshrq_n_u8(qlb, 4)),
                                                    vreinterpretq_s8_u8(vshlq_n_u8(h3, 4))), k32);
                    vst1q_s8(dst + 0 + l0,  v0);
                    vst1q_s8(dst + 32 + l0, v1);
                    vst1q_s8(dst + 64 + l0, v2);
                    vst1q_s8(dst + 96 + l0, v3);
                }
                qlp += 64; qhp += 32;
            }
        }

        for (int t = 0; t < n; t++) {
            const int8_t *xp = X + (size_t)t * cols + b * 256;
            float acc = Y[(size_t)t * ldY];
            const float *ap = asc ? asc + (size_t)t * (cols / 32 + 1) + b * 8 : NULL;
            for (int half = 0; half < 2; half++) {
                const int8_t *wb = w + half * 128;
                const int8_t *xb = xp + half * 128;
                const int8_t *scb = sc + half * 8;
                // 四个位置段，各 32 长；段内 scale 每 16 一档
                for (int h = 0; h < 2; h++) {
                    for (int seg = 0; seg < 4; seg++) {
                        const int off = seg * 32 + h * 16;
                        const int32_t dt = vaddvq_s32(
                            vdotq_s32(vdupq_n_s32(0), vld1q_s8(wb + off), vld1q_s8(xb + off)));
                        const float am = ap ? ap[half * 4 + seg] : 1.0f;
                        acc += (float)dt * d * scb[seg * 2 + h] * am;
                    }
                }
            }
            Y[(size_t)t * ldY] = acc;
        }
    }
}

// ── Q5_K 融合 GEMV ──────────────────────────────────────
// 和 Q4_K 一样是非对称量化，但要额外从 qh 补第 5 位。
// 直接照搬 q4k_dot_bs 的结构，只多一步"加 16"。
float q5k_dot_bs(const void *raw, const int8_t *x, const int16_t *bs, size_t n) {
    const uint8_t *p = (const uint8_t *)raw;
    size_t nblk = n / 256;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;

    const uint8x16_t mask4 = vdupq_n_u8(0x0F);
    const int8x16_t k16 = vdupq_n_s8(16);

    for (size_t b = 0; b < nblk; b++, p += 176) {
        uint16_t dh, dmh;
        memcpy(&dh, p, 2);
        memcpy(&dmh, p + 2, 2);
        const float d = half2f_fast(dh);
        const float dmin = half2f_fast(dmh);
        const uint8_t *scales = p + 4;
        const uint8_t *qh = p + 16;
        const uint8_t *qs = p + 48;
        const int8_t *xp = x + b * 256;

        float ds[8], dm[8];
        for (int i = 0; i < 8; i++) {
            uint8_t sc, mn;
            get_scale_min_k4(i, scales, &sc, &mn);
            ds[i] = d * sc;
            dm[i] = dmin * mn;
        }
        const int16_t *bp = bs ? bs + b * 16 : NULL;

        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            const int si = j / 32;
            // qh 一共 32 字节，覆盖全部 256 个权重 —— 每轮 j 用的是不同的
            // 位掩码（u1/u2 左移），指针不动。官方实现里也是 `qh[l]` 直接索引。
            const uint8_t *qhseg = qh;

            int32x4_t accLo = vdupq_n_s32(0), accHi = vdupq_n_s32(0);

            for (int g = 0; g < 2; g++) {
                const uint8x16_t packed = vld1q_u8(qs + g * 16);
                const uint8x16_t hiBits = vld1q_u8(qhseg + g * 16);

                // 低 4 位 + 第 5 位
                int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(packed, mask4));
                const uint8x16_t m1 = vtstq_u8(hiBits, vdupq_n_u8(u1));
                lo = vaddq_s8(lo, vandq_s8(vreinterpretq_s8_u8(m1), k16));

                // 高 4 位 + 第 5 位
                int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(packed, 4));
                const uint8x16_t m2 = vtstq_u8(hiBits, vdupq_n_u8(u2));
                hi = vaddq_s8(hi, vandq_s8(vreinterpretq_s8_u8(m2), k16));

                accLo = vdotq_s32(accLo, lo, vld1q_s8(xp + j + g * 16));
                accHi = vdotq_s32(accHi, hi, vld1q_s8(xp + j + 32 + g * 16));
            }

            int32_t sumLo, sumHi;
            if (bp) {
                sumLo = bp[(si + 0) * 2] + bp[(si + 0) * 2 + 1];
                sumHi = bp[(si + 1) * 2] + bp[(si + 1) * 2 + 1];
            } else {
                sumLo = sumHi = 0;
                for (int k = 0; k < 32; k++) sumLo += xp[j + k];
                for (int k = 0; k < 32; k++) sumHi += xp[j + 32 + k];
            }

            const float v0 = (float)vaddvq_s32(accLo) * ds[si + 0] - (float)sumLo * dm[si + 0];
            const float v1 = (float)vaddvq_s32(accHi) * ds[si + 1] - (float)sumHi * dm[si + 1];
            switch ((j / 64) & 3) {
            case 0: acc0 += v0; acc1 += v1; break;
            case 1: acc2 += v0; acc3 += v1; break;
            case 2: acc0 += v0; acc2 += v1; break;
            default: acc1 += v0; acc3 += v1; break;
            }

            qs += 32;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    return (acc0 + acc1) + (acc2 + acc3);
}

// Q5_K 批量（结构同 q4k_gemm_row，多一步从 qh 补第 5 位）
void q6k_gemm_row(const void *raw, const int8_t *X, float *Y,
                  int n, size_t cols, size_t ldY) {
    q6k_gemm_row_a(raw, X, NULL, Y, n, cols, ldY);
}
void q5k_gemm_row(const void *raw, const int8_t *X, const int16_t *bs,
                  float *Y, int n, size_t cols, size_t ldY) {
    const uint8_t *p = (const uint8_t *)raw;
    const size_t nblk = cols / 256;
    const uint8x16_t mask4 = vdupq_n_u8(0x0F);
    const int8x16_t  k16   = vdupq_n_s8(16);

    for (int t = 0; t < n; t++) Y[(size_t)t * ldY] = 0.f;

    for (size_t b = 0; b < nblk; b++, p += 176) {
        uint16_t dh, dmh;
        memcpy(&dh, p, 2);
        memcpy(&dmh, p + 2, 2);
        const float d = half2f_fast(dh);
        const float dmin = half2f_fast(dmh);
        const uint8_t *scales = p + 4;
        const uint8_t *qh = p + 16;
        const uint8_t *qs = p + 48;

        float ds[8], dm[8];
        for (int i = 0; i < 8; i++) {
            uint8_t sc, mn;
            get_scale_min_k4(i, scales, &sc, &mn);
            ds[i] = d * sc;
            dm[i] = dmin * mn;
        }

        // 解出 8 个子块（每块 32 个 int8，已含第 5 位）
        alignas(16) int8_t w[8][32];
        {
            uint8_t u1 = 1, u2 = 2;
            for (int seg = 0; seg < 4; seg++) {
                const int si = seg * 2;
                const uint8_t *q = qs + seg * 32;
                for (int g = 0; g < 2; g++) {
                    const uint8x16_t a  = vld1q_u8(q + g * 16);
                    const uint8x16_t hb = vld1q_u8(qh + g * 16);
                    int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(a, mask4));
                    lo = vaddq_s8(lo, vandq_s8(vreinterpretq_s8_u8(
                            vtstq_u8(hb, vdupq_n_u8(u1))), k16));
                    int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(a, 4));
                    hi = vaddq_s8(hi, vandq_s8(vreinterpretq_s8_u8(
                            vtstq_u8(hb, vdupq_n_u8(u2))), k16));
                    vst1q_s8(w[si]     + g * 16, lo);
                    vst1q_s8(w[si + 1] + g * 16, hi);
                }
                u1 <<= 2;
                u2 <<= 2;
            }
        }

        for (int t = 0; t < n; t++) {
            const int8_t *xp = X + (size_t)t * cols + b * 256;
            const int16_t *bp = bs ? bs + (size_t)t * (cols / 16 + 1) + b * 16 : NULL;
            const float *ap = NULL; /* q5k 暂不支持 per-32 激活 scale */
            float acc = Y[(size_t)t * ldY];

            for (int si = 0; si < 8; si++) {
                const int8_t *ws = w[si];
                const int8_t *xs = xp + si * 32;
                int32x4_t va = vdotq_s32(vdupq_n_s32(0), vld1q_s8(ws), vld1q_s8(xs));
                va = vdotq_s32(va, vld1q_s8(ws + 16), vld1q_s8(xs + 16));
                const int32_t dt = vaddvq_s32(va);

                int32_t ssum;
                if (bp) ssum = bp[si * 2] + bp[si * 2 + 1];
                else { ssum = 0; for (int k2 = 0; k2 < 32; k2++) ssum += xs[k2]; }
                const float am = 1.0f;
                acc += (float)dt * ds[si] * am - (float)ssum * dm[si] * am;
            }
            Y[(size_t)t * ldY] = acc;
        }
    }
}

// ═══════════════════════════════════════════════════════
// per-block 激活量化的内核（精度版）
//
// 【和 q4k_dot_bs 的区别】原来整个激活向量共用一个 scale，
// 反量化时在 super-block 层统一乘。现在每 32 个元素一个 scale，
// 必须在**子块**层乘：acc += int_dot * (d*sc) * act_scale[si]。
//
// 【为什么权重 scale 和激活 scale 要一起乘】分开乘就是每个子块
// 两次乘法 + 一次加法，融合成一个乘数只需一次。代价是每个子块
// 要先算 d*sc*act_scale —— 但那也是个乘法，总数不变，指令更少。

float q4k_dot_bsb(const void *raw, const int8_t *x, const float *as,
                  const int16_t *bs, size_t n) {
    const uint8_t *p = (const uint8_t *)raw;
    size_t nblk = n / 256;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    const uint8x16_t mask4 = vdupq_n_u8(0x0F);

    for (size_t b = 0; b < nblk; b++, p += 144) {
        uint16_t dh, dmh;
        memcpy(&dh, p, 2);
        memcpy(&dmh, p + 2, 2);
        const float d = half2f_fast(dh);
        const float dmin = half2f_fast(dmh);
        const uint8_t *scales = p + 4;
        const uint8_t *qs = p + 16;
        const int8_t *xp = x + b * 256;
        const float *asp = as + b * 8;
        const int16_t *bp = bs ? bs + b * 16 : NULL;

        float ds[8], dm[8];
        for (int i = 0; i < 8; i++) {
            uint8_t sc, mn;
            get_scale_min_k4(i, scales, &sc, &mn);
            ds[i] = d * sc * asp[i];
            dm[i] = dmin * mn * asp[i];
        }

        for (int j = 0; j < 256; j += 64) {
            const int si = j / 32;
            int32x4_t accLo = vdupq_n_s32(0), accHi = vdupq_n_s32(0);

            for (int g = 0; g < 2; g++) {
                const uint8x16_t packed = vld1q_u8(qs + g * 16);
                const int8x16_t xlo = vld1q_s8(xp + j + g * 16);
                const int8x16_t xhi = vld1q_s8(xp + j + 32 + g * 16);
                accLo = vdotq_s32(accLo, vreinterpretq_s8_u8(vandq_u8(packed, mask4)), xlo);
                accHi = vdotq_s32(accHi, vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)), xhi);
            }

            int32_t sumLo, sumHi;
            if (bp) {
                sumLo = bp[(si + 0) * 2] + bp[(si + 0) * 2 + 1];
                sumHi = bp[(si + 1) * 2] + bp[(si + 1) * 2 + 1];
            } else {
                sumLo = sumHi = 0;
                for (int k = 0; k < 32; k++) sumLo += xp[j + k];
                for (int k = 0; k < 32; k++) sumHi += xp[j + 32 + k];
            }

            const float v0 = (float)vaddvq_s32(accLo) * ds[si + 0] - (float)sumLo * dm[si + 0];
            const float v1 = (float)vaddvq_s32(accHi) * ds[si + 1] - (float)sumHi * dm[si + 1];
            switch ((j / 64) & 3) {
            case 0: acc0 += v0; acc1 += v1; break;
            case 1: acc2 += v0; acc3 += v1; break;
            case 2: acc0 += v0; acc2 += v1; break;
            default: acc1 += v0; acc3 += v1; break;
            }
            qs += 32;
        }
    }
    return (acc0 + acc1) + (acc2 + acc3);
}

float q5k_dot_bsb(const void *raw, const int8_t *x, const float *as,
                  const int16_t *bs, size_t n) {
    const uint8_t *p = (const uint8_t *)raw;
    size_t nblk = n / 256;
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    const uint8x16_t mask4 = vdupq_n_u8(0x0F);
    const int8x16_t  k16   = vdupq_n_s8(16);

    for (size_t b = 0; b < nblk; b++, p += 176) {
        uint16_t dh, dmh;
        memcpy(&dh, p, 2);
        memcpy(&dmh, p + 2, 2);
        const float d = half2f_fast(dh);
        const float dmin = half2f_fast(dmh);
        const uint8_t *scales = p + 4;
        const uint8_t *qh = p + 16;
        const uint8_t *qs = p + 48;
        const int8_t *xp = x + b * 256;
        const float *asp = as + b * 8;
        const int16_t *bp = bs ? bs + b * 16 : NULL;

        float ds[8], dm[8];
        for (int i = 0; i < 8; i++) {
            uint8_t sc, mn;
            get_scale_min_k4(i, scales, &sc, &mn);
            ds[i] = d * sc * asp[i];
            dm[i] = dmin * mn * asp[i];
        }

        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            const int si = j / 32;
            int32x4_t accLo = vdupq_n_s32(0), accHi = vdupq_n_s32(0);
            for (int g = 0; g < 2; g++) {
                const uint8x16_t packed = vld1q_u8(qs + g * 16);
                const uint8x16_t hb = vld1q_u8(qh + g * 16);
                int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(packed, mask4));
                lo = vaddq_s8(lo, vandq_s8(vreinterpretq_s8_u8(vtstq_u8(hb, vdupq_n_u8(u1))), k16));
                int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(packed, 4));
                hi = vaddq_s8(hi, vandq_s8(vreinterpretq_s8_u8(vtstq_u8(hb, vdupq_n_u8(u2))), k16));
                accLo = vdotq_s32(accLo, lo, vld1q_s8(xp + j + g * 16));
                accHi = vdotq_s32(accHi, hi, vld1q_s8(xp + j + 32 + g * 16));
            }
            int32_t sumLo, sumHi;
            if (bp) {
                sumLo = bp[(si + 0) * 2] + bp[(si + 0) * 2 + 1];
                sumHi = bp[(si + 1) * 2] + bp[(si + 1) * 2 + 1];
            } else {
                sumLo = sumHi = 0;
                for (int k = 0; k < 32; k++) sumLo += xp[j + k];
                for (int k = 0; k < 32; k++) sumHi += xp[j + 32 + k];
            }
            const float v0 = (float)vaddvq_s32(accLo) * ds[si + 0] - (float)sumLo * dm[si + 0];
            const float v1 = (float)vaddvq_s32(accHi) * ds[si + 1] - (float)sumHi * dm[si + 1];
            switch ((j / 64) & 3) {
            case 0: acc0 += v0; acc1 += v1; break;
            case 1: acc2 += v0; acc3 += v1; break;
            case 2: acc0 += v0; acc2 += v1; break;
            default: acc1 += v0; acc3 += v1; break;
            }
            qs += 32;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    return (acc0 + acc1) + (acc2 + acc3);
}

float q6k_dot_b(const void *raw, const int8_t *x, const float *as, size_t n) {
    const uint8_t *p = (const uint8_t *)raw;
    size_t nblk = n / 256;
    float acc0 = 0.f, acc1 = 0.f;
    const uint8x16_t m0F = vdupq_n_u8(0x0F);
    const uint8x16_t m03 = vdupq_n_u8(0x03);
    const int8x16_t  k32 = vdupq_n_s8(32);

    for (size_t b = 0; b < nblk; b++, p += 210) {
        uint16_t dh;
        memcpy(&dh, p + 208, 2);
        const float d = half2f_fast(dh);
        const uint8_t *ql = p;
        const uint8_t *qh = p + 128;
        const int8_t *sc = (const int8_t *)(p + 192);
        const int8_t *xp = x + b * 256;
        const float *asp = as + b * 8;

        for (int half = 0; half < 2; half++) {
            const int8_t *xb = xp + half * 128;
            for (int h = 0; h < 2; h++) {
                const int l0 = h * 16;
                const uint8x16_t qla = vld1q_u8(ql + l0);
                const uint8x16_t qlb = vld1q_u8(ql + 32 + l0);
                const uint8x16_t qhv = vld1q_u8(qh + l0);
                const uint8x16_t h0 = vandq_u8(qhv, m03);
                const uint8x16_t h1 = vandq_u8(vshrq_n_u8(qhv, 2), m03);
                const uint8x16_t h2 = vandq_u8(vshrq_n_u8(qhv, 4), m03);
                const uint8x16_t h3 = vshrq_n_u8(qhv, 6);
                // 【保留 vsubq】试过用 bsums 补偿掉这 4 次减法，
                // 但多算 4 次 vaddlvq_s8 更贵（3.5 → 3.82 GB/s），已回退。
                int8x16_t v0 = vsubq_s8(vaddq_s8(vreinterpretq_s8_u8(vandq_u8(qla, m0F)),
                                                vreinterpretq_s8_u8(vshlq_n_u8(h0, 4))), k32);
                int8x16_t v1 = vsubq_s8(vaddq_s8(vreinterpretq_s8_u8(vandq_u8(qlb, m0F)),
                                                vreinterpretq_s8_u8(vshlq_n_u8(h1, 4))), k32);
                int8x16_t v2 = vsubq_s8(vaddq_s8(vreinterpretq_s8_u8(vshrq_n_u8(qla, 4)),
                                                vreinterpretq_s8_u8(vshlq_n_u8(h2, 4))), k32);
                int8x16_t v3 = vsubq_s8(vaddq_s8(vreinterpretq_s8_u8(vshrq_n_u8(qlb, 4)),
                                                vreinterpretq_s8_u8(vshlq_n_u8(h3, 4))), k32);
                // 【先发 4 个 vdot，再统一横向求和】
                // 原来每算一个就 vaddvq_s32，等于强制串行 —— vaddvq 有 5~7 cycle 延迟，
                // 四次串起来就是 20+ cycle 的停顿。分开写让 CPU 能交错发射。
                const int32x4_t t0 = vdotq_s32(vdupq_n_s32(0), v0, vld1q_s8(xb + 0 + l0));
                const int32x4_t t1 = vdotq_s32(vdupq_n_s32(0), v1, vld1q_s8(xb + 32 + l0));
                const int32x4_t t2 = vdotq_s32(vdupq_n_s32(0), v2, vld1q_s8(xb + 64 + l0));
                const int32x4_t t3 = vdotq_s32(vdupq_n_s32(0), v3, vld1q_s8(xb + 96 + l0));
                const int32_t s0 = vaddvq_s32(t0);
                const int32_t s1 = vaddvq_s32(t1);
                const int32_t s2 = vaddvq_s32(t2);
                const int32_t s3 = vaddvq_s32(t3);

                // 激活组号 = half*4 + seg（和 h 无关，h 只决定块内高低 16 位）。
                // 四个 s0..s3 分别对应 seg=0..3，各用自己那一组的激活 scale。
                const int gbase = half * 4;
                const float a0 = asp[gbase + 0], a1 = asp[gbase + 1];
                const float a2 = asp[gbase + 2], a3 = asp[gbase + 3];
                if (h == 0) {
                    acc0 += (float)s0 * d * a0 * sc[0];
                    acc0 += (float)s1 * d * a1 * sc[2];
                    acc0 += (float)s2 * d * a2 * sc[4];
                    acc0 += (float)s3 * d * a3 * sc[6];
                } else {
                    acc1 += (float)s0 * d * a0 * sc[1];
                    acc1 += (float)s1 * d * a1 * sc[3];
                    acc1 += (float)s2 * d * a2 * sc[5];
                    acc1 += (float)s3 * d * a3 * sc[7];
                }
            }
            ql += 64; qh += 32; sc += 8;
        }
    }
    return acc0 + acc1;
}

// Q2_K：2-bit 值域 0..3，非对称（有 min），结构 256 权重/84 字节。
//
// 【两个性能坑，都在这版改掉】
// 1. 原版每 16 个值都 vst1q + vld1q 走一趟栈 —— 改成寄存器内直接 SDOT。
// 2. 原版用标量循环算 sum(x)（16 次加法）—— 改为传入预计算的 bsums。
//    （实测 Q2_K 的张量效率只有 29%，而 Q4_K 是 94%，差距全在这两处）
float q2k_dot_bb(const void *raw, const int8_t *x, const float *as,
                 const int16_t *bs, size_t n) {
    const uint8_t *p = (const uint8_t *)raw;
    size_t nblk = n / 256;
    float acc = 0.f;
    const uint8x16_t m3 = vdupq_n_u8(3);

    for (size_t b = 0; b < nblk; b++, p += 84) {
        const uint8_t *sc = p;
        const uint8_t *qs = p + 16;
        uint16_t dh, dmh;
        memcpy(&dh, p + 80, 2);
        memcpy(&dmh, p + 82, 2);
        const float d = half2f_fast(dh);
        const float dmin = half2f_fast(dmh);
        const int8_t *xp = x + b * 256;
        const float *asp = as + b * 8;
        const int16_t *bp = bs ? bs + b * 16 : NULL;

        int is = 0, w = 0;
        for (int half = 0; half < 2; half++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                const int si = half * 4 + j;
                const uint8_t s0 = sc[is++];
                const uint8_t s1 = sc[is++];
                // 一次处理 32 个值（两个 16 字节段），寄存器内解 2-bit
                const uint8x16_t v0 = vld1q_u8(qs);
                const uint8x16_t v1 = vld1q_u8(qs + 16);
                const uint8x16_t q0 = vandq_u8(vshlq_u8(v0, vdupq_n_s8((int8_t)(-shift))), m3);
                const uint8x16_t q1 = vandq_u8(vshlq_u8(v1, vdupq_n_s8((int8_t)(-shift))), m3);

                const int32_t dt0 = vaddvq_s32(
                    vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(q0), vld1q_s8(xp + w)));
                const int32_t dt1 = vaddvq_s32(
                    vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(q1), vld1q_s8(xp + w + 16)));

                int32_t sum0, sum1;
                if (bp) {
                    // bsums 是每 16 个一个，这里正好对应
                    const int bi = w / 16;
                    sum0 = bp[bi];
                    sum1 = bp[bi + 1];
                } else {
                    sum0 = 0; for (int k = 0; k < 16; k++) sum0 += xp[w + k];
                    sum1 = 0; for (int k = 0; k < 16; k++) sum1 += xp[w + 16 + k];
                }
                const float a_s = asp[si];
                acc += (float)dt0 * (d * (s0 & 0xF) * a_s) - (float)sum0 * (dmin * (s0 >> 4) * a_s);
                acc += (float)dt1 * (d * (s1 & 0xF) * a_s) - (float)sum1 * (dmin * (s1 >> 4) * a_s);
                w += 32;
                shift += 2;
            }
            qs += 32;
        }
    }
    return acc;
}

float q3k_dot_b(const void *raw, const int8_t *x, const float *as, size_t n) {
    const uint8_t *p = (const uint8_t *)raw;
    size_t nblk = n / 256;
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    float acc = 0.f;
    for (size_t b = 0; b < nblk; b++, p += 110) {
        const uint8_t *hm = p;
        const uint8_t *qs = p + 32;
        uint16_t dh;
        memcpy(&dh, p + 108, 2);
        const float d_all = half2f_fast(dh);
        uint32_t aux[4];
        memcpy(aux, p + 96, 12);
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t *scales = (const int8_t *)aux;
        const int8_t *xp = x + b * 256;
        const float *asp = as + b * 8;

        uint8_t m = 1;
        int is = 0, w = 0;
        for (int half = 0; half < 2; half++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                const int si = half * 4 + j;
                for (int seg = 0; seg < 2; seg++) {
                    alignas(16) int8_t t[16];
                    const uint8x16_t v = vld1q_u8(qs + seg * 16);
                    const uint8x16_t lo = vandq_u8(vshlq_u8(v, vdupq_n_s8((int8_t)(-shift))), vdupq_n_u8(3));
                    const uint8x16_t hi = vld1q_u8(hm + seg * 16);
                    const uint8x16_t notset = vtstq_u8(hi, vdupq_n_u8(m));
                    int8x16_t val = vreinterpretq_s8_u8(lo);
                    val = vsubq_s8(val, vandq_s8(vmvnq_u8(notset), vdupq_n_s8(4)));
                    vst1q_s8(t, val);
                    const int32_t dt = vaddvq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(t), vld1q_s8(xp + w)));
                    acc += (float)dt * (d_all * (scales[is] - 32) * asp[si]);
                    is++;
                    w += 16;
                }
                shift += 2;
                m <<= 1;
            }
            qs += 32;
        }
    }
    return acc;
}
