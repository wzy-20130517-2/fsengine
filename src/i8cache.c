#include "i8cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void dequant_q6k_row(const void *raw, float *out, size_t n);

typedef struct {
    char name[64];
    int8_t *w8;       // [rows][cols] int8
    float  *scales;   // [rows][cols/256] float
    size_t rows, cols;
} Entry;

static Entry *g_e = NULL;
static int g_n = 0, g_cap = 0;

const int8_t *i8_lookup(const char *nm, const float **scales, size_t *cols) {
    for (int i = 0; i < g_n; i++)
        if (strcmp(g_e[i].name, nm) == 0) {
            if (scales) *scales = g_e[i].scales;
            if (cols) *cols = g_e[i].cols;
            return g_e[i].w8;
        }
    return NULL;
}

// 只缓存 Q6_K 的大张量（attn_qkv 443MB + token_embd 262MB，占 33% 权重）
static int want(const GgufTensor *t) {
    if (t->type != 14) return 0;
    return strstr(t->name, "attn_qkv") || strstr(t->name, "token_embd");
}

int i8_init(GgufModel *gm) {
    if (getenv("I8_DISABLE")) return 0;
    g_cap = 16;
    g_e = calloc(g_cap, sizeof(Entry));

    for (int t = 0; t < gm->n_tensors; t++) {
        GgufTensor *ts = &gm->tensors[t];
        if (!want(ts)) continue;
        const size_t cols = ts->dims[0], rows = ts->dims[1];
        if (cols % 256) continue;
        const size_t rb = cols / 256 * 210;
        const size_t nb = cols / 256;

        int8_t *w8 = malloc(rows * cols);
        float  *sc = malloc(rows * nb * sizeof(float));
        float  *tmp = malloc(cols * sizeof(float));
        if (!w8 || !sc || !tmp) { free(w8); free(sc); free(tmp); continue; }

        for (size_t r = 0; r < rows; r++) {
            dequant_q6k_row((const uint8_t *)ts->data + r * rb, tmp, cols);
            for (size_t b = 0; b < nb; b++) {
                float mx = 0.f;
                for (int i = 0; i < 256; i++) {
                    float a = fabsf(tmp[b * 256 + i]);
                    if (a > mx) mx = a;
                }
                const float s = (mx > 1e-8f) ? mx / 127.f : 1.f;
                sc[r * nb + b] = s;
                const float inv = 1.f / s;
                for (int i = 0; i < 256; i++) {
                    float q = tmp[b * 256 + i] * inv;
                    q = q > 127.f ? 127.f : (q < -127.f ? -127.f : q);
                    w8[r * cols + b * 256 + i] = (int8_t)lrintf(q);
                }
            }
        }
        free(tmp);

        if (g_n >= g_cap) { g_cap *= 2; g_e = realloc(g_e, g_cap * sizeof(Entry)); }
        snprintf(g_e[g_n].name, sizeof g_e[g_n].name, "%s", ts->name);
        g_e[g_n].w8 = w8; g_e[g_n].scales = sc;
        g_e[g_n].rows = rows; g_e[g_n].cols = cols;
        g_n++;
    }
    return g_n;
}

void i8_free(void) {
    for (int i = 0; i < g_n; i++) { free(g_e[i].w8); free(g_e[i].scales); }
    free(g_e); g_e = NULL; g_n = g_cap = 0;
}
