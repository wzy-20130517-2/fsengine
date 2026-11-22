// GGUF 解析器：只读、mmap、不拷贝权重。
//
// 【为什么不拷贝】
// 模型 2.5GB，手机总共 11GB 内存且已用 6.6GB。把权重读进 malloc 的堆
// 会让常驻内存翻倍，还会因 page cache 和匿名页并存而挤爆。
// mmap 让权重直接以文件页的形式存在，内核按需调入、内存紧张时回收，
// 多个进程可以共享同一份物理页。这是手机端必须的做法。
//
// 【为什么自己写而不复用 ggml】
// 目标是"为手机做的引擎"，ggml 的抽象层（张量描述、后端调度、量化表）
// 有大量分支判断和间接调用，在 A77 这种没有分支预测优势的乱序核上
// 每一层间接都是一次预测失败。这里只支持我们需要的量化类型，
// 解量化路径可以在编译期全部内联。
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

// ── 读取游标 ─────────────────────────────────────────────
typedef struct {
    const uint8_t *base;
    size_t size;
    size_t pos;
    int err;
} Reader;

static void rd_need(Reader *r, size_t n) {
    if (r->pos + n > r->size) r->err = 1;
}

static uint8_t rd_u8(Reader *r) {
    rd_need(r, 1);
    if (r->err) return 0;
    return r->base[r->pos++];
}

static uint32_t rd_u32(Reader *r) {
    rd_need(r, 4);
    if (r->err) return 0;
    uint32_t v;
    memcpy(&v, r->base + r->pos, 4);
    r->pos += 4;
    return v;
}

static uint64_t rd_u64(Reader *r) {
    rd_need(r, 8);
    if (r->err) return 0;
    uint64_t v;
    memcpy(&v, r->base + r->pos, 8);
    r->pos += 8;
    return v;
}

// GGUF 的字符串：uint64 长度 + 字节（不含结尾 0）
// 返回 malloc 出来的 C 字符串，调用方负责 free（只用于元数据键名，量很少）
static char *rd_str(Reader *r) {
    uint64_t n = rd_u64(r);
    if (r->err || n > (1u << 28)) { r->err = 1; return NULL; }
    rd_need(r, n);
    if (r->err) return NULL;
    char *s = malloc(n + 1);
    if (!s) { r->err = 1; return NULL; }
    memcpy(s, r->base + r->pos, n);
    s[n] = 0;
    r->pos += n;
    return s;
}

// 跳过一段元数据值（只需要知道它的总字节数，不关心内容）
static void rd_skip_value(Reader *r, uint32_t type, int depth) {
    if (depth > 8 || r->err) { r->err = 1; return; }
    switch (type) {
    case 0: case 1: case 7: r->pos += 1; rd_need(r, 0); break;   // u8/i8/bool
    case 2: case 3: r->pos += 2; break;                          // u16/i16
    case 4: case 5: case 6: r->pos += 4; break;                  // u32/i32/f32
    case 10: case 11: case 12: r->pos += 8; break;               // u64/i64/f64
    case 8: { char *s = rd_str(r); free(s); break; }              // string
    case 9: {                                                    // array
        uint32_t at = rd_u32(r);
        uint64_t n = rd_u64(r);
        if (r->err || n > (1u << 28)) { r->err = 1; return; }
        // 定长类型批量跳过，省掉 n 次函数调用
        size_t elem = 0;
        switch (at) {
        case 0: case 1: case 7: elem = 1; break;
        case 2: case 3: elem = 2; break;
        case 4: case 5: case 6: elem = 4; break;
        case 10: case 11: case 12: elem = 8; break;
        }
        if (elem) { rd_need(r, (size_t)n * elem); r->pos += (size_t)n * elem; }
        else for (uint64_t i = 0; i < n && !r->err; i++) rd_skip_value(r, at, depth + 1);
        break;
    }
    default: r->err = 1;
    }
}

// ── 元数据查找 ───────────────────────────────────────────
// 只保留我们实际要用的键，其余跳过 —— 避免为 131072 个 token 分配内存。
typedef struct {
    char *key;
    uint32_t type;
    uint64_t u64;
    double f64;
    char *str;
} MetaEntry;

typedef struct {
    MetaEntry *e;
    int n, cap;
} MetaTable;

static int meta_find(MetaTable *t, const char *key) {
    for (int i = 0; i < t->n; i++)
        if (strcmp(t->e[i].key, key) == 0) return i;
    return -1;
}

static uint64_t meta_u64(MetaTable *t, const char *key, uint64_t def) {
    int i = meta_find(t, key);
    if (i < 0) return def;
    MetaEntry *m = &t->e[i];
    if (m->type == 4 || m->type == 5) return m->u64;         // u32 / i32
    if (m->type == 10 || m->type == 11) return m->u64;       // u64 / i64
    if (m->type == 0 || m->type == 1) return m->u64;
    return def;
}

static double meta_f64(MetaTable *t, const char *key, double def) {
    int i = meta_find(t, key);
    if (i < 0) return def;
    MetaEntry *m = &t->e[i];
    if (m->type == 6 || m->type == 12) return m->f64;
    if (m->type == 4 || m->type == 5 || m->type == 10 || m->type == 11)
        return (double)m->u64;
    return def;
}

static const char *meta_str(MetaTable *t, const char *key) {
    int i = meta_find(t, key);
    return i < 0 ? NULL : t->e[i].str;
}

// 我们要保留的键前缀 —— 其余全部跳过以节省内存和时间
static int key_wanted(const char *k) {
    static const char *want[] = {
        "general.architecture", "general.name",
        "spark2_5.", "qwen3.", "qwen2.", "llama.",
        "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id", "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token", "tokenizer.ggml.model",
        NULL,
    };
    for (int i = 0; want[i]; i++)
        if (strncmp(k, want[i], strlen(want[i])) == 0) return 1;
    return 0;
}

// ── 主入口 ───────────────────────────────────────────────
GgufModel *gguf_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return NULL; }

    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return NULL; }

    void *map = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) { perror("mmap"); return NULL; }

    Reader r = { (const uint8_t *)map, (size_t)st.st_size, 0, 0 };

    // magic "GGUF"
    uint32_t magic = rd_u32(&r);
    if (magic != 0x46554747) {          // 'G','G','U','F' 小端读成 u32
        fprintf(stderr, "不是 GGUF 文件（magic=%08x）\n", magic);
        munmap(map, st.st_size);
        return NULL;
    }
    uint32_t ver = rd_u32(&r);
    uint64_t n_tensors = rd_u64(&r);
    uint64_t n_kv = rd_u64(&r);
    if (ver < 2 || ver > 3) {
        fprintf(stderr, "不支持的 GGUF 版本 %u\n", ver);
        munmap(map, st.st_size);
        return NULL;
    }

    // ── 元数据 ──
    MetaTable mt = { NULL, 0, 0 };
    mt.cap = 64;
    mt.e = calloc(mt.cap, sizeof(MetaEntry));

    for (uint64_t i = 0; i < n_kv && !r.err; i++) {
        char *key = rd_str(&r);
        if (!key) break;
        uint32_t type = rd_u32(&r);

        if (!key_wanted(key)) {
            rd_skip_value(&r, type, 0);
            free(key);
            continue;
        }
        if (mt.n >= mt.cap) {
            mt.cap *= 2;
            mt.e = realloc(mt.e, mt.cap * sizeof(MetaEntry));
        }
        MetaEntry *m = &mt.e[mt.n++];
        memset(m, 0, sizeof(*m));
        m->key = key;
        m->type = type;

        switch (type) {
        case 0: case 1: case 7: m->u64 = rd_u8(&r); break;
        case 2: case 3: { rd_need(&r, 2); m->u64 = (uint64_t)r.base[r.pos]; r.pos += 2; break; }
        case 4: case 5: m->u64 = rd_u32(&r); break;
        case 6: { uint32_t v = rd_u32(&r); float f; memcpy(&f, &v, 4); m->f64 = f; break; }
        case 10: case 11: m->u64 = rd_u64(&r); break;
        case 12: { uint64_t v = rd_u64(&r); double d; memcpy(&d, &v, 8); m->f64 = d; break; }
        case 8: m->str = rd_str(&r); break;
        default: rd_skip_value(&r, type, 0); break;
        }
    }

    // ── 张量信息 ──
    GgufTensor *ts = calloc(n_tensors, sizeof(GgufTensor));
    for (uint64_t i = 0; i < n_tensors && !r.err; i++) {
        ts[i].name = rd_str(&r);
        uint32_t nd = rd_u32(&r);
        if (nd > 4) { r.err = 1; break; }
        ts[i].n_dims = (int)nd;
        for (uint32_t d = 0; d < nd; d++)
            ts[i].dims[d] = rd_u64(&r);
        ts[i].type = rd_u32(&r);
        ts[i].offset = rd_u64(&r);
    }

    // 权重数据紧跟在张量信息之后，按 alignment 对齐
    uint64_t align = meta_u64(&mt, "general.alignment", 32);
    if (align == 0) align = 32;
    size_t data_start = (r.pos + align - 1) / align * align;

    if (r.err) {
        fprintf(stderr, "GGUF 解析失败（pos=%zu / %zu）\n", r.pos, r.size);
        munmap(map, st.st_size);
        return NULL;
    }

    GgufModel *m = calloc(1, sizeof(GgufModel));
    m->path = strdup(path);
    m->map = map;
    m->map_size = (size_t)st.st_size;
    m->data = (const uint8_t *)map + data_start;
    m->data_size = (size_t)st.st_size - data_start;
    m->tensors = ts;
    m->n_tensors = (int)n_tensors;

    // 主动预读整个文件到 page cache。实测收益不明显（文件本来就常在 cache 里），
    // 但对冷启动有帮助，成本只是一次系统调用。
    madvise(map, st.st_size, MADV_WILLNEED);

    // 【大页】2.5GB 模型按 4KB 页算是 64 万个页表项，TLB 装不下，
    // 每次访问都可能缺页遍历页表。MADV_HUGEPAGE 让内核尽量合并成 2MB 大页，
    // 页表项降到 1250 个，TLB 命中率大增。
    // 注意：需要系统开了 THP（/sys/kernel/mm/transparent_hugepage/enabled 不能是 never）。
    madvise(map, st.st_size, MADV_HUGEPAGE);


    // 把张量指针直接指到 mmap 上
    for (int i = 0; i < m->n_tensors; i++) {
        if (ts[i].offset > m->data_size) { ts[i].data = NULL; continue; }
        ts[i].data = m->data + ts[i].offset;
    }

    // ── 抄出配置 ──
    const char *arch = meta_str(&mt, "general.architecture");
    m->arch = arch ? strdup(arch) : strdup("unknown");

    char k[128];
#define M_U64(field, suffix, def) \
    snprintf(k, sizeof k, "%s.%s", m->arch, suffix); \
    m->field = meta_u64(&mt, k, def)
#define M_F64(field, suffix, def) \
    snprintf(k, sizeof k, "%s.%s", m->arch, suffix); \
    m->field = meta_f64(&mt, k, def)

    M_U64(n_layer, "block_count", 0);
    M_U64(n_embd, "embedding_length", 0);
    M_U64(n_ff, "feed_forward_length", 0);
    M_U64(n_head, "attention.head_count", 0);
    M_U64(n_head_kv, "attention.head_count_kv", 0);
    M_U64(vocab_size, "vocab_size", 0);
    M_U64(n_ctx, "context_length", 0);
    M_U64(rope_dims, "rope.dimension_count", 0);
    M_U64(window, "attention.sliding_window", 0);
    M_F64(rms_eps, "attention.layer_norm_rms_epsilon", 1e-6);
    M_F64(rope_theta, "rope.freq_base", 10000.0);

    // 可选：head_dim 不直接给，得从 qkv 权重形状推
    m->head_dim = 0;
    m->bos_id = (int)meta_u64(&mt, "tokenizer.ggml.bos_token_id", 0);
    m->eos_id = (int)meta_u64(&mt, "tokenizer.ggml.eos_token_id", 1);
#undef M_U64
#undef M_F64

    // 从 attn_qkv 的形状推 head_dim：
    // 融合权重是 [n_embd, (n_head + 2*n_head_kv) * head_dim]
    // 例如 2560 x 6144，16+2*4=24, 6144/24 = 256
    for (int i = 0; i < m->n_tensors; i++) {
        if (strstr(ts[i].name, "attn_qkv.weight")) {
            if (ts[i].n_dims == 2) {
                uint64_t cols = ts[i].dims[1];
                uint64_t per = m->n_head + 2 * m->n_head_kv;
                if (per) m->head_dim = (int)(cols / per);
            }
            break;
        }
    }

    // 清理元数据表（张量名已抄进 GgufTensor，不再需要）
    for (int i = 0; i < mt.n; i++) { free(mt.e[i].key); free(mt.e[i].str); }
    free(mt.e);

    return m;
}

GgufTensor *gguf_find(GgufModel *m, const char *name) {
    for (int i = 0; i < m->n_tensors; i++)
        if (strcmp(m->tensors[i].name, name) == 0) return &m->tensors[i];
    return NULL;
}

void gguf_close(GgufModel *m) {
    if (!m) return;
    for (int i = 0; i < m->n_tensors; i++) free(m->tensors[i].name);
    free(m->tensors);
    free(m->path);
    free(m->arch);
    munmap(m->map, m->map_size);
    free(m);
}

// 每种量化类型每 256 个权重占多少字节
size_t ggml_type_size(GgmlType t) {
    switch (t) {
    case GGML_TYPE_F32:  return 1024;
    case GGML_TYPE_F16:  return 512;
    case GGML_TYPE_Q4_0: return 144;
    case GGML_TYPE_Q4_1: return 160;
    case GGML_TYPE_Q5_0: return 176;
    case GGML_TYPE_Q5_1: return 192;
    case GGML_TYPE_Q8_0: return 272;
    case GGML_TYPE_Q2_K: return 84;
    case GGML_TYPE_Q3_K: return 110;
    case GGML_TYPE_Q4_K: return 144;
    case GGML_TYPE_Q5_K: return 176;
    case GGML_TYPE_Q6_K: return 210;
    default: return 0;
    }
}

// 一个"块"里有多少个权重
size_t ggml_blck_size(GgmlType t) {
    switch (t) {
    case GGML_TYPE_F32: case GGML_TYPE_F16: return 1;
    case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0: return 32;
    case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K: return 256;
    default: return 0;
    }
}

size_t gguf_tensor_bytes(const GgufTensor *t) {
    size_t n = 1;
    for (int i = 0; i < t->n_dims; i++) n *= t->dims[i];
    size_t bs = ggml_blck_size((GgmlType)t->type);
    size_t ts = ggml_type_size((GgmlType)t->type);
    if (!bs || !ts) return 0;
    return n / bs * ts;
}

void gguf_summary(const GgufModel *m) {
    printf("arch        %s\n", m->arch);
    printf("layers      %llu\n", (unsigned long long)m->n_layer);
    printf("hidden      %llu\n", (unsigned long long)m->n_embd);
    printf("ffn         %llu\n", (unsigned long long)m->n_ff);
    printf("heads       %llu Q / %llu KV, head_dim %d\n",
           (unsigned long long)m->n_head,
           (unsigned long long)m->n_head_kv, m->head_dim);
    printf("vocab       %llu\n", (unsigned long long)m->vocab_size);
    printf("ctx         %llu\n", (unsigned long long)m->n_ctx);
    printf("window      %llu\n", (unsigned long long)m->window);
    printf("rope        %llu dims, theta %.0f, eps %.1e\n",
           (unsigned long long)m->rope_dims, m->rope_theta, m->rms_eps);
    printf("tensors     %d\n", m->n_tensors);
    printf("file        %.1f MB\n", m->map_size / 1048576.0);
}