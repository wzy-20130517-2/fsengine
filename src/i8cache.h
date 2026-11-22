#ifndef FSENGINE_I8CACHE_H
#define FSENGINE_I8CACHE_H
#include "gguf.h"
#include <stddef.h>

// Q6_K 张量的 int8 预解包缓存。
//
// 【为什么值得】Q6_K 每 256 个权重要 4 次取位（ql 低 4 位 + qh 高 2 位拼接），
// 实测只有 4.74 GB/s；预解包成 int8 后直接 vdot，能到 11.26 GB/s（1.95 倍）。
// 代价：210 字节/256 值 → 256 字节/256 值，只多占 22%。
int  i8_init(GgufModel *gm);
void i8_free(void);
// 查到 int8 数据；scales 是每 256 个权重的 float scale，返回 NULL 表示没有
const int8_t *i8_lookup(const char *name, const float **scales, size_t *cols);

#endif
