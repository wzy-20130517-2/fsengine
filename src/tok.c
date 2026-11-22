// 从 GGUF 里读出 tokenizer（BPE），实现 encode。
//
// 【为什么单独解析】gguf.c 的元数据是流式读的，读完就丢，只保留白名单里的
// 几个超参。tokenizer.ggml.tokens 有 131072 项、merges 有 130716 项，不适合
// 塞进 GgufModel。这里重新扫一遍文件头，只取这两张表。
//
// 【字节映射】Spark 的词表是 GPT-2 风格：token 里存的是 byte→unicode 映射后的
// 字符串（空格写成 U+2581 那种），不是原始字节。encode 时必须把输入文本按
// 同样的映射拆成单字节单元，再做 BPE merge。
#include "tok.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct { const uint8_t *base; size_t pos, size; int err; } R;
static void need(R *r, size_t n){ if (r->pos + n > r->size) r->err = 1; }
static uint8_t  ru8 (R *r){ need(r,1); return r->err?0:r->base[r->pos++]; }
static uint32_t ru32(R *r){ need(r,4); uint32_t v; memcpy(&v,r->base+r->pos,4); r->pos+=4; return r->err?0:v; }
static uint64_t ru64(R *r){ need(r,8); uint64_t v; memcpy(&v,r->base+r->pos,8); r->pos+=8; return r->err?0:v; }
static const char *rstr(R *r, uint64_t *len){
    uint64_t n = ru64(r); *len = n;
    if (r->err || r->pos + n > r->size) { r->err = 1; return ""; }
    const char *p = (const char *)(r->base + r->pos); r->pos += n; return p;
}

// 跳过任意类型的值
static void skip_val(R *r, uint32_t t){
    uint64_t n;
    switch (t) {
    case 0: case 1: case 7: ru8(r); break;
    case 2: case 3: need(r,2); r->pos+=2; break;
    case 4: case 5: case 6: ru32(r); break;
    case 10: case 11: case 12: ru64(r); break;
    case 8: rstr(r,&n); break;
    case 9: {
        uint32_t et = ru32(r); uint64_t cnt = ru64(r);
        for (uint64_t i = 0; i < cnt && !r->err; i++) skip_val(r, et);
        break;
    }
    default: r->err = 1; break;
    }
}

// byte → unicode 的逆映射（GPT-2 用）
// 256 个字节里有 188 个 printable 直接映射到自己，剩下 68 个映射到 256..323
int uni_to_byte(uint32_t cp){
    // U+2581（LOWER ONE EIGHTH BLOCK）是 SentencePiece/GPT-2 词表里表示空格的符号
    if (cp == 0x2581) return ' ';
    if (cp < 256) {
        // 直接映射区：'!'..'~'、'¡'..'¬'、'®'..'ÿ'
        if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255))
            return (int)cp;
        return -1;
    }
    // 256..323 对应被挤出的 68 个字节
    int b = -1;
    static int tab_built = 0;
    static int16_t tab[68];
    if (!tab_built) {
        int k = 0;
        int used[256] = {0};
        for (int i = 33; i <= 126; i++) used[i] = 1;
        for (int i = 161; i <= 172; i++) used[i] = 1;
        for (int i = 174; i <= 255; i++) used[i] = 1;
        for (int i = 0; i < 256; i++) if (!used[i]) tab[k++] = (int16_t)i;
        tab_built = 1;
    }
    if (cp >= 256 && cp < 256 + 68) b = tab[cp - 256];
    return b;
}

// UTF-8 解码一个码点
int utf8_next(const char **pp, uint32_t *cp){
    const uint8_t *p = (const uint8_t *)*pp;
    if (!p[0]) return 0;
    if (p[0] < 0x80) { *cp = p[0]; *pp += 1; return 1; }
    if ((p[0] & 0xE0) == 0xC0 && (p[1] & 0xC0) == 0x80) {
        *cp = ((p[0] & 0x1F) << 6) | (p[1] & 0x3F); *pp += 2; return 1;
    }
    if ((p[0] & 0xF0) == 0xE0 && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80) {
        *cp = ((p[0] & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F); *pp += 3; return 1;
    }
    if ((p[0] & 0xF8) == 0xF0 && (p[1] & 0xC0) == 0x80 && (p[2] & 0xC0) == 0x80 && (p[3] & 0xC0) == 0x80) {
        *cp = ((p[0] & 0x07) << 18) | ((p[1] & 0x3F) << 12) | ((p[2] & 0x3F) << 6) | (p[3] & 0x3F);
        *pp += 4; return 1;
    }
    *pp += 1; *cp = 0xFFFD; return 1;
}

typedef struct { char *s; int len; } Piece;

// qsort 的比较器需要拿到 Tokenizer，用文件级指针传
static Tokenizer *g_sort_tk;
static int tok_cmp_qsort(const void *pa, const void *pb) {
    return tok_cmp(g_sort_tk, *(const int *)pa, *(const int *)pb);
}

int tok_load(const char *gguf_path, Tokenizer *tk) {
    FILE *f = fopen(gguf_path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc((size_t)sz);
    if (!buf) { fclose(f); return 0; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return 0; }
    fclose(f);

    R r = { buf, 0, (size_t)sz, 0 };
    uint32_t magic = ru32(&r);
    if (magic != 0x46554747u) { free(buf); return 0; }
    ru32(&r);                       // version
    uint64_t n_tensors = ru64(&r), n_kv = ru64(&r);
    (void)n_tensors;

    // 先把 token 字符串原样存下来（还没做 byte 映射）
    char **raw = NULL; uint64_t n_raw = 0;
    char **mg  = NULL; uint64_t n_mg  = 0;

    for (uint64_t i = 0; i < n_kv && !r.err; i++) {
        uint64_t klen; const char *key = rstr(&r, &klen);
        uint32_t t = ru32(&r);
        if (t != 9) { skip_val(&r, t); continue; }
        uint32_t et = ru32(&r); uint64_t cnt = ru64(&r);
        if (!strncmp(key, "tokenizer.ggml.token_type", klen)) {
            tk->tok_type = calloc(cnt, sizeof(int8_t)); tk->n_type = (int)cnt;
            for (uint64_t j = 0; j < cnt && !r.err; j++) tk->tok_type[j] = (int8_t)(int32_t)ru32(&r);
        } else if (!strncmp(key, "tokenizer.ggml.tokens", klen)) {
            raw = calloc(cnt, sizeof(char*)); n_raw = cnt;
            for (uint64_t j = 0; j < cnt && !r.err; j++) {
                uint64_t sl; const char *s = rstr(&r, &sl);
                raw[j] = strndup(s, sl);
            }
        } else if (!strncmp(key, "tokenizer.ggml.merges", klen)) {
            mg = calloc(cnt, sizeof(char*)); n_mg = cnt;
            for (uint64_t j = 0; j < cnt && !r.err; j++) {
                uint64_t sl; const char *s = rstr(&r, &sl);
                mg[j] = strndup(s, sl);
            }
        } else {
            for (uint64_t j = 0; j < cnt && !r.err; j++) skip_val(&r, et);
        }
    }
    free(buf);
    if (!raw || !n_raw) { free(raw); free(mg); return 0; }

    tk->n_tok = (int)n_raw;
    tk->tok_str = calloc(n_raw, sizeof(char*));   // 按字节串存的 token
    tk->tok_len = calloc(n_raw, sizeof(int));
    tk->tok_id  = calloc(n_raw, sizeof(int));     // 排序后用于查找的 id
    tk->n_merge = (int)n_mg;
    tk->merge_a = calloc(n_mg ? n_mg : 1, sizeof(int));
    tk->merge_b = calloc(n_mg ? n_mg : 1, sizeof(int));

    // token 字符串 → 原始字节串（反转 byte→unicode 映射）
    int bad = 0;
    for (uint64_t i = 0; i < n_raw; i++) {
        // 【特殊 token 不能做字节映射】<｜start▁of▁sentence｜> 里的 ｜(U+FF5C)
        // 和 ▁(U+2581) 映射不到单字节，强行映射会退化成只剩 '<'
        // （之前 id 0-2 全变成 '<' 就是这个原因）。token_type != 1 按原文保留。
        if (tk->tok_type && (int)i < tk->n_type && tk->tok_type[i] != 1) {
            tk->tok_str[i] = strdup(raw[i]);
            tk->tok_len[i] = (int)strlen(raw[i]);
            tk->tok_id[i] = (int)i;
            free(raw[i]);
            continue;
        }
        const char *p = raw[i];
        char *out = malloc(strlen(raw[i]) + 4);
        int olen = 0;
        while (*p) {
            uint32_t cp;
            if (!utf8_next(&p, &cp)) break;
            int b = uni_to_byte(cp);
            if (b < 0) { bad = 1; break; }
            out[olen++] = (char)(uint8_t)b;
        }
        tk->tok_str[i] = out;
        tk->tok_len[i] = olen;
        tk->tok_id[i] = (int)i;
        free(raw[i]);
    }
    free(raw);
    if (bad) { fprintf(stderr, "[tok] 词表里有无法映射的码点\n"); }

    // merge 规则："A B" → (bytes(A), bytes(B))，记顺序号
    for (uint64_t i = 0; i < n_mg; i++) {
        char *sp = strchr(mg[i], ' ');
        if (!sp) { tk->merge_a[i] = tk->merge_b[i] = -1; free(mg[i]); continue; }
        *sp = 0;
        // 同样做反映射，存成「该字符串在词表里的 id」以省比较
        char a[64], b2[64]; int al = 0, bl = 0;
        for (const char *p = mg[i]; *p && al < 60; ) { uint32_t cp; utf8_next(&p,&cp); a[al++] = (char)uni_to_byte(cp); }
        for (const char *p = sp+1; *p && bl < 60; ) { uint32_t cp; utf8_next(&p,&cp); b2[bl++] = (char)uni_to_byte(cp); }
        tk->merge_a[i] = tok_lookup(tk, a, al);
        tk->merge_b[i] = tok_lookup(tk, b2, bl);
        free(mg[i]);
    }
    free(mg);

    // 排序索引，供二分查找。13 万项必须用 qsort —— 插入排序是 O(n²)，跑不完。
    g_sort_tk = tk;
    qsort(tk->tok_id, (size_t)tk->n_tok, sizeof(int), tok_cmp_qsort);
    fprintf(stderr, "[tok] 词表 %d, merge %d\n", tk->n_tok, tk->n_merge);
    return 1;
}

void tok_free(Tokenizer *tk) {
    if (!tk) return;
    if (tk->tok_str) for (int i = 0; i < tk->n_tok; i++) free(tk->tok_str[i]);
    free(tk->tok_str); free(tk->tok_len); free(tk->tok_id);
    free(tk->merge_a); free(tk->merge_b); free(tk->tok_type);
}
