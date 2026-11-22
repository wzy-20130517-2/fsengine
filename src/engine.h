#ifndef FSENGINE_ENGINE_H
#define FSENGINE_ENGINE_H
#include "gguf.h"

typedef struct {
    GgufModel *gm;
    int H, NH, NKV, HD, FF, L, V;
    int max_ctx;
    int rope_dims_swa, rope_base_full;
    int pos;

    float *x, *h, *qkv, *attn, *gate, *ff1, *ff2, *logits;
    int8_t *qa;      // 激活量化缓冲
    int16_t *bsa;    // bsums（每 16 个 int8 的和）
    float  *asc;     // per-block 激活 scale（每 32 个一组）
    // x8 路径专用：per-256 量化的激活（和 per-32 那套并存，各取所需）
    int8_t *qa8;     // per-256 量化的激活
    float  *asc8;    // per-256 的 scale（每 256 个一个）
    int16_t *bsa8;   // per-256 的 bsums
    __fp16 **kcache; // [L][max_ctx][NKV*HD]
    __fp16 **vcache;

    // 【预转换的 norm 权重】原来每层 3 次 malloc + tensor_to_f32（共 108 次/step），
    // 这些权重是固定的，开引擎时转一次就够。
    float *norm_attn;   // [L][H]
    float *norm_ffn;    // [L][H]
    float *norm_out;    // [H]  最后那个 output_norm

    // 【预解析张量指针】原来每层 8 次 gguf_find（线性搜 290 个张量 + strcmp），
    // 36 层就是 288 次搜索 / 8.4 万次字符串比较每 token。
    GgufTensor **t_qkv, **t_attn_out, **t_ffn_gate, **t_ffn_up, **t_ffn_down, **t_attn_gate;
    GgufTensor *t_emb;
} Engine;

Engine *engine_open(const char *path, int max_ctx);
// 指定线程数的版本（线程池会被多个 Engine 共用，所以只 init 一次）
Engine *engine_open_t(const char *path, int max_ctx, int nthreads);
void    engine_close(Engine *e);
float  *engine_step(Engine *e, int token);
// 批量前向：一次处理 n 个位置，返回 [n][V] 的 logits（调用方释放）
// 投机解码验证用它 —— 一次前向确认 k 个 draft token
float  *engine_step_batch(Engine *e, const int *tokens, int n);

#endif
