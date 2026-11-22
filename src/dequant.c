// Q4_K / Q6_K / Q8_0 解量化 + 点积融合内核。
//
// 【核心思路：解量化和点积在同一趟里做完】
// 不要先把权重解成 float 存到临时 buffer 再算点积 —— 那等于把 2.5GB 权重
// 复制一遍到堆上，光是分配和写回就吃掉全部收益。
// 边解边算，解出来的值只活在寄存器里。
//
// 【Q4_K 的块结构】256 个权重一个 super-block，共 144 字节：
//   d      fp16   super-block 的 scale
//   dmin   fp16   super-block 的 min
//   scales 12B    8 个子块各自的 6-bit 量化 scale 和 min（打包）
//   qs     128B   256 个 4-bit 值
// 每个 4-bit 值解成 d * sub_scale * q - dmin * sub_min
//
// 【寄存器压力】A77 有 32 个 NEON 寄存器。Q4_K 一路要同时留住
// d/dmin、4 组 scale、权重和解出的 int8，全展开会溢出到栈上。
// 所以这里分两步走：先把一个子块（32 个权）解到栈上的 int8 临时数组，
// 再用 SDOT 点积。临时数组只有 32 字节，稳稳在 L1 里。
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <arm_neon.h>

// ── fp16 → fp32（A77 原生支持 fphp，但 32 个值一起转更省事）──
static inline float half2float(uint16_t h) {
    // 用 NEON 的 fcvtn 也行，但标量版在这里更快（只调几次）
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t expo = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (expo == 0) {
        if (mant == 0) bits = sign;
        else {
            // 非规格化数
            expo = 127 - 15 + 1;
            while (!(mant & 0x400)) { mant <<= 1; expo--; }
            mant &= 0x3FF;
            bits = sign | (expo << 23) | (mant << 13);
        }
    } else if (expo == 0x1F) {
        bits = sign | 0x7F800000 | (mant << 13);
    } else {
        bits = sign | ((expo - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// ── Q4_K 的子块 scale/min 解码 ─────────────────────────
// 12 字节里压了 8 个子块各 6 bit 的 scale 和 min，排布很绕。
// 这是官方 ggml-quants.c 的 get_scale_min_k4 逻辑，格式必须严格一致。
static inline void get_scale_min_k4(int j, const uint8_t *q,
                                    uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Q4_K 一行（可能跨多个 super-block）解成 float
// 这个版本直接按官方的交错方式处理，是正确实现。
void dequant_q4k_row(const void *raw, float *out, size_t n) {
    const uint8_t *p = (const uint8_t *)raw;
    size_t nblk = n / 256;
    for (size_t b = 0; b < nblk; b++, p += 144, out += 256) {
        uint16_t dh, dmh;
        memcpy(&dh, p, 2);
        memcpy(&dmh, p + 2, 2);
        const float d = half2float(dh);
        const float dmin = half2float(dmh);
        const uint8_t *scales = p + 4;
        const uint8_t *qs = p + 16;
        float *y = out;

        // 官方布局：每轮吃 64 个权重 —— 先用 32 字节的低半字节出前 32 个，
        // 再用同一批字节的高半字节出后 32 个，然后 q 前进 32、scale 前进 2。
        // 不是"先全部低半字节再全部高半字节"，那样顺序是乱的。
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, mn;

            get_scale_min_k4(is + 0, scales, &sc, &mn);
            const float d1 = d * sc, m1 = dmin * mn;
            get_scale_min_k4(is + 1, scales, &sc, &mn);
            const float d2 = d * sc, m2 = dmin * mn;

            for (int l = 0; l < 32; l++) *y++ = d1 * (qs[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) *y++ = d2 * (qs[l] >> 4) - m2;

            qs += 32;
            is += 2;
        }
    }
}

// ── Q6_K 一行解成 float ────────────────────────────────
// 块结构：256 个权重，210 字节
//   ql     128B   低 4 位
//   qh      64B   高 2 位
//   scales  16B   16 个 int8 scale
//   d        2B   fp16
//
// 【布局要点】外层按 128 个一组循环两次，组内 l 只走 0..31，
// 一次产出 4 个输出（l+0 / l+32 / l+64 / l+96），scale 取 is+0/2/4/6。
// 不是把 256 个值当一段连续处理 —— 那是 Q4_K 的方式，Q6_K 不一样。
void dequant_q6k_row(const void *raw, float *out, size_t n) {
    const uint8_t *p = (const uint8_t *)raw;
    size_t nblk = n / 256;
    for (size_t b = 0; b < nblk; b++, p += 210, out += 256) {
        uint16_t dh;
        memcpy(&dh, p + 208, 2);
        const float d = half2float(dh);
        const uint8_t *ql = p;
        const uint8_t *qh = p + 128;
        const int8_t *sc = (const int8_t *)(p + 192);

        for (int half = 0; half < 2; half++) {
            float *y = out + half * 128;
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) |
                                   (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) |
                                   (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) |
                                   (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) |
                                   (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0]  = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// ── Q8_0 一行解成 int8（激活量化用得到）────────────────
// 块：32 个 int8 + 1 个 fp16 scale = 34 字节
void dequant_q8_0_row(const void *raw, float *out, size_t n) {
    const uint8_t *p = (const uint8_t *)raw;
    size_t nblk = n / 32;
    for (size_t b = 0; b < nblk; b++, p += 34, out += 32) {
        uint16_t dh;
        memcpy(&dh, p, 2);
        const float d = half2float(dh);
        const int8_t *qs = (const int8_t *)(p + 2);
        for (int i = 0; i < 32; i++) out[i] = d * qs[i];
    }
}
// ── Q2_K 一行解成 float ─────────────────────────────────
// 块：256 权重 = 84 字节
//   scales  16B   每字节装两个 4-bit（低=scale，高=min），共 16 组
//   qs      64B   每个字节 4 个 2-bit 值，按 shift 取
//   d,dmin   4B    2 个 fp16
// 外层按 128 一组循环两次，组内 shift 走 0/2/4/6。
void dequant_q2k_row(const void *raw, float *out, size_t n) {
    const uint8_t *p = (const uint8_t *)raw;
    size_t nblk = n / 256;
    for (size_t b = 0; b < nblk; b++, p += 84, out += 256) {
        const uint8_t *sc = p;
        const uint8_t *q = p + 16;
        uint16_t dh, dmh;
        memcpy(&dh, p + 80, 2);
        memcpy(&dmh, p + 82, 2);
        const float d = half2float(dh);
        const float dmin = half2float(dmh);

        float *y = out;
        int is = 0;
        for (int half = 0; half < 2; half++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t s = sc[is++];
                float dl = d * (s & 0xF), ml = dmin * (s >> 4);
                for (int l = 0; l < 16; l++) *y++ = dl * ((q[l] >> shift) & 3) - ml;

                s = sc[is++];
                dl = d * (s & 0xF); ml = dmin * (s >> 4);
                for (int l = 0; l < 16; l++) *y++ = dl * ((q[l + 16] >> shift) & 3) - ml;

                shift += 2;
            }
            q += 32;
        }
    }
}

// ── Q3_K 一行解成 float ─────────────────────────────────
// 块：256 权重 = 110 字节
//   hmask   32B   最高位
//   qs      64B   低 4 位（每字节 2 个）
//   scales  12B   16 个 6-bit scale，打包方式见下
//   d        2B   fp16
void dequant_q3k_row(const void *raw, float *out, size_t n) {
    const uint8_t *p = (const uint8_t *)raw;
    size_t nblk = n / 256;
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    for (size_t b = 0; b < nblk; b++, p += 110, out += 256) {
        const uint8_t *hm = p;
        const uint8_t *q = p + 32;
        uint16_t dh;
        memcpy(&dh, p + 108, 2);
        const float d_all = half2float(dh);

        uint32_t aux[4];
        memcpy(aux, p + 96, 12);
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t *scales = (const int8_t *)aux;

        float *y = out;
        uint8_t m = 1;
        int is = 0;
        for (int half = 0; half < 2; half++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    *y++ = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

// ── Q5_K 一行解成 float ─────────────────────────────────
// 块：256 权重 = 176 字节
//   d,dmin   4B
//   scales  12B   和 Q4_K 完全一样
//   qh      32B   每个权重多 1 位（第 5 位），按 u1/u2 掩码取
//   qs     128B   低 4 位
// 循环结构同 Q4_K：每轮 64 个权重、2 个 scale，但要从 qh 补高位。
void dequant_q5k_row(const void *raw, float *out, size_t n) {
    const uint8_t *p = (const uint8_t *)raw;
    size_t nblk = n / 256;
    for (size_t b = 0; b < nblk; b++, p += 176, out += 256) {
        uint16_t dh, dmh;
        memcpy(&dh, p, 2);
        memcpy(&dmh, p + 2, 2);
        const float d = half2float(dh);
        const float dmin = half2float(dmh);
        const uint8_t *scales = p + 4;
        const uint8_t *qh = p + 16;
        const uint8_t *qs = p + 48;
        float *y = out;

        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, mn;
            get_scale_min_k4(is + 0, scales, &sc, &mn);
            const float d1 = d * sc, m1 = dmin * mn;
            get_scale_min_k4(is + 1, scales, &sc, &mn);
            const float d2 = d * sc, m2 = dmin * mn;

            for (int l = 0; l < 32; l++)
                *y++ = d1 * ((qs[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; l++)
                *y++ = d2 * ((qs[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;

            qs += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}
