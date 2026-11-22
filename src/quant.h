// 激活量化（per-32-block，对齐 llama.cpp 的 Q8_K 思路）
#ifndef FSENGINE_QUANT_H
#define FSENGINE_QUANT_H

#include <stdint.h>
#include <stddef.h>

// 把 n 个 float 量化成 int8，每 32 个一组独立 scale。
// q:      输出 n 个 int8
// scales: 输出 n/32 个 float（每组一个）
// bs:     输出 n/16 个 int16（每 16 个的和，Q4_K/Q5_K 的 min 项要用），可为 NULL
void quant_act_block(const float *x, int8_t *q, float *scales, int16_t *bs, int n);

#endif
