#ifndef FSENGINE_TOK_H
#define FSENGINE_TOK_H
#include <stddef.h>
#include <stdint.h>

typedef struct {
    int   n_tok;
    char **tok_str;   // 每个 token 对应的原始字节串（已反转 byte→unicode 映射）
    int  *tok_len;
    int  *tok_id;     // 按字节串排序后的 id 索引，用于二分查找
    int   n_merge;
    int  *merge_a;    // 每条 merge 的左右两半（存 token id）
    int  *merge_b;
    int8_t *tok_type;   // 1=normal 2=unknown 3=control 4=user_defined 5=unused 6=byte
    int   n_type;
} Tokenizer;

int  tok_load(const char *gguf_path, Tokenizer *tk);
void tok_free(Tokenizer *tk);
int  tok_lookup(Tokenizer *tk, const char *s, int len);
int  tok_cmp(Tokenizer *tk, int a, int b);

// encode：把 UTF-8 文本转成 token id 序列。返回写入个数。
// 先按 GPT-2 的预切分规则拆词（空格/标点/中文字），再对每段做 BPE merge。
int  tok_encode(Tokenizer *tk, const char *text, int *out, int max_out);

// decode：单个 token 的原始字节串
const char *tok_piece(Tokenizer *tk, int id, int *len);

int  utf8_next(const char **pp, uint32_t *cp);
int  uni_to_byte(uint32_t cp);

#endif
