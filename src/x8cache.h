#ifndef FSENGINE_X8CACHE_H
#define FSENGINE_X8CACHE_H
#include "gguf.h"
#include <stddef.h>

// 重排缓存：把 Q4_K 张量按 8 行交错重排，存在堆里。
// 只对最热的张量做（ffn_gate / ffn_up），内存多占 ~1GB。
int  x8_init(GgufModel *gm, int nthreads);
void x8_free(void);
// 查到重排后的数据，没有返回 NULL
const void *x8_lookup(const char *tensor_name, size_t *rowbytes);
// 这个张量是否走了 x8（决定用哪套激活量化）
int  x8_has(const char *tensor_name);
// 重排后张量的原始量化类型（12=Q4_K, 14=Q6_K），-1 表示没重排
int  x8_type(const char *tensor_name);
// x8 功能是否启用
int  x8_enabled(void);

#endif
