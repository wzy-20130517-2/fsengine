// GGUF 数据结构与接口
#ifndef FSENGINE_GGUF_H
#define FSENGINE_GGUF_H

#include <stdint.h>
#include <stddef.h>

// 量化类型编号 —— 对齐官方 ggml.h 的 ggml_type
typedef enum {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
} GgmlType;

typedef struct {
    char    *name;
    int      n_dims;
    uint64_t dims[4];
    uint32_t type;
    uint64_t offset;
    const uint8_t *data;   // 直接指向 mmap 区域
    // x8 缓存标记（0=没有，12=Q4_Kx8，14=Q6_Kx8）。
    // 用字段而不是每次 x8_has() 查表 —— 原来每层要 3 次线性查找 × 65 个 strcmp，
    // 36 层就是 7000 次字符串比较，纯浪费。
    int      x8_type;
    // x8 重排后的指针和行字节数，重排时直接写回这里。
    // 原来每次 GEMV 都要 x8_lookup() 线性查 65 个条目（gate/up/down/qkv/out × 36 层
    // = 每 token 180 次查找 × 65 个 strcmp），纯浪费。
    const void *x8_ptr;
    size_t      x8_rb;
} GgufTensor;

typedef struct {
    char    *path;
    void    *map;
    size_t   map_size;
    const uint8_t *data;
    size_t   data_size;

    GgufTensor *tensors;
    int      n_tensors;

    // 从元数据抄出来的配置
    char    *arch;
    uint64_t n_layer;
    uint64_t n_embd;
    uint64_t n_ff;
    uint64_t n_head;
    uint64_t n_head_kv;
    int      head_dim;
    uint64_t vocab_size;
    uint64_t n_ctx;
    uint64_t window;
    uint64_t rope_dims;
    double   rope_theta;
    double   rms_eps;
    int      bos_id;
    int      eos_id;
} GgufModel;

GgufModel *gguf_open(const char *path);
void       gguf_close(GgufModel *m);
GgufTensor *gguf_find(GgufModel *m, const char *name);
size_t     gguf_tensor_bytes(const GgufTensor *t);
size_t     ggml_type_size(GgmlType t);
size_t     ggml_blck_size(GgmlType t);
void       gguf_summary(const GgufModel *m);

#endif