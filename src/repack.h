#ifndef FSENGINE_REPACK_H
#define FSENGINE_REPACK_H
#include <stdint.h>
#include <stddef.h>

typedef struct {
    uint16_t d[8];
    uint16_t dmin[8];
    uint8_t  scales[96];
    uint8_t  qs[1024];
} blk_q4Kx8;

// 把 nrows 行的 Q4_K 权重重排成 8 路交错布局。
// out 需要 nrows * rowbytes 字节（布局变了但总大小不变）。
void repack_q4k_rows(const void *raw, int nrows, size_t rowbytes, void *out);
// Q6_K 的 8 路重排（一个 super-block 1680 字节 = 8×210）
void repack_q6k_rows(const void *raw, int nrows, size_t rowbytes, void *out);
// Q6_Kx8 内核：一次算 8 行
void q6k_x8_block(const void *blkp, const int8_t *x, const float *as, float *out8);
// Q4_Kx8 内核：一次算 8 行、1 个位置
void q4k_x8_block(const blk_q4Kx8 *blk, const int8_t *x, const float *as,
                  const int16_t *bs, float *out8);
// Q4_Kx8 多位置版：权重只读一次，同时算 nb 个位置（nb<=4）
// x 按 nb 份布局，每份 256 字节；as/bs 同理；out8 每份 8 个 float
void q4k_x8_block_n(const blk_q4Kx8 *blk, const int8_t *x, size_t xstride,
                    const float *as, const int16_t *bs, float *out8, int nb);

#endif
