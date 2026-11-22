// Spark2.5 完整前向：decode 单 token 路径。
//
// 架构（从 mlx-vlm 官方实现 + GGUF 元数据核实）：
//   x = emb(tok)
//   for L in 36:
//     h = x + Attn(RMSNorm(x))
//     x = h + MLP(RMSNorm(h))          # MLP 是 GeGLU: down(gelu(gate(x)) * up(x))
//   logits = lm_head(RMSNorm(x))       # lm_head 与 token_embd 绑定
//
// Attn 的特殊之处：
//   - 融合 QKV 一次投出 [q(4096) | k(1024) | v(1024)]
//   - 16 Q heads / 4 KV heads（GQA 4x），head_dim 256
//   - **每头一个 sigmoid gate**（g_proj: 2560 -> 16），乘在注意力输出上
//   - 层类型 [S,S,S,F] 循环：S=sliding(窗口512)，F=full
//   - partial RoPE：S 层转满 256 维 θ=1e4，F 层只转 64 维 θ=5e6
#include "engine.h"
#include <time.h>
double g_t_qkv, g_t_attn, g_t_gate, g_t_down, g_t_norm, g_t_rope, g_t_geglu, g_t_quant;
static double _ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static double _tick_a[8];
enum { TK_QKV=0, TK_ATTN, TK_GATE, TK_DOWN, TK_ROPE, TK_QUANT, TK_NORM, TK_GEGLU, TK_OTHER, TK_N };
double g_tk[TK_N];
static double _tk_now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static double _tk_a;
#define TSTA() (_tk_a = _tk_now())
#define TEND(i) (g_tk[i] += _tk_now() - _tk_a)
#define TICK(i) (_tick_a[i]=_ns())
#define TOCK(i) (g_t_tmp[i] += _ns()-_tick_a[i])
double g_t_tmp[8];
#include "tensor.h"
#include "pool.h"
#include "x8cache.h"
#include "i8cache.h"
extern void quant_act_q8k(const float *, int8_t *, float *, int16_t *, int);
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <arm_neon.h>


// ── 小工具 ───────────────────────────────────────────────
// 【4 路独立累加】原来是一条 acc 链串行 FMA，2560 维 = 640 次迭代，
// FMA 的 4~5 cycle 延迟会完全暴露。拆成 4 条链后延迟hide住。
static void rmsnorm(const float *x, const float *w, float *out, int n, float eps) {
    float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
    float32x4_t a2 = vdupq_n_f32(0.f), a3 = vdupq_n_f32(0.f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const float32x4_t v0 = vld1q_f32(x + i);
        const float32x4_t v1 = vld1q_f32(x + i + 4);
        const float32x4_t v2 = vld1q_f32(x + i + 8);
        const float32x4_t v3 = vld1q_f32(x + i + 12);
        a0 = vfmaq_f32(a0, v0, v0);
        a1 = vfmaq_f32(a1, v1, v1);
        a2 = vfmaq_f32(a2, v2, v2);
        a3 = vfmaq_f32(a3, v3, v3);
    }
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; i < n; i++) s += x[i] * x[i];
    const float r = 1.0f / sqrtf(s / (float)n + eps);
    const float32x4_t vr = vdupq_n_f32(r);
    for (i = 0; i + 16 <= n; i += 16) {
        vst1q_f32(out + i,      vmulq_f32(vmulq_f32(vld1q_f32(x + i),      vr), vld1q_f32(w + i)));
        vst1q_f32(out + i + 4,  vmulq_f32(vmulq_f32(vld1q_f32(x + i + 4),  vr), vld1q_f32(w + i + 4)));
        vst1q_f32(out + i + 8,  vmulq_f32(vmulq_f32(vld1q_f32(x + i + 8),  vr), vld1q_f32(w + i + 8)));
        vst1q_f32(out + i + 12, vmulq_f32(vmulq_f32(vld1q_f32(x + i + 12), vr), vld1q_f32(w + i + 12)));
    }
    for (; i < n; i++) out[i] = x[i] * r * w[i];
}

// 激活量化：float -> int8，同时算 bsums（Q4_K 的 min 项要用）
// 用对称量化：scale = max|x| / 127
// per-block 量化：每 32 个一组 scale。返回 NULL（scales 写入 asc）
static void quantize_act_b(const float *x, int8_t *q, float *asc, int16_t *bs, int n) {
    extern void quant_act_block(const float *, int8_t *, float *, int16_t *, int);
    extern void quant_act_q8k(const float *, int8_t *, float *, int16_t *, int);
    static int use_q8k = -1;
    if (use_q8k < 0) use_q8k = getenv("USE_Q8K") != NULL;
    if (use_q8k && (n % 256) == 0) { quant_act_q8k(x, q, asc, bs, n); return; }
    quant_act_block(x, q, asc, bs, n);
}

static float quantize_act(const float *x, int8_t *q, int16_t *bs, int n) {
    float mx = 0.f;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > mx) mx = a; }
    const float scale = (mx > 1e-8f) ? mx / 127.0f : 1.0f;
    const float inv = 1.0f / scale;
    for (int i = 0; i < n; i++) {
        float v = x[i] * inv;
        v = v > 127.f ? 127.f : (v < -127.f ? -127.f : v);
        q[i] = (int8_t)lrintf(v);
    }
    if (bs) {
        int o = 0;
        for (int i = 0; i + 16 <= n; i += 16, o++) {
            float s = 0.f;
            for (int k = 0; k < 16; k++) s += (float)q[i + k];
            bs[o] = (int16_t)s;
        }
    }
    return scale;
}

// partial RoPE：只转前 rope_dims 维，traditional=False（即 GPT-NeoX 风格）
// 位置 pos 用 inv_freq[i] = base^(-2i/rope_dims)
// RoPE。频率只依赖 (i, rope_dims, base)，和 head 无关 ——
// 原来每个 head 都重算一遍 powf/cosf/sinf，16 个 head 就是 16 倍浪费。
// 实测 128 个频率对算一次 2.2us，36 层累计 79us/token；
// 按 head 逐个算是 16 倍，即 1.27ms/token，占非 GEMV 开销的 3.6%。
// 预计算的 inv_freq 表：[0]=SWA(θ=1e4, rdim=256)、[1]=full(θ=5e6, rdim=64)
// powf(base, -2i/rope_dims) 与 pos 无关，原来每层每次调用都重算 128 次 powf
// （36 层 × q/k 两次 = 9216 次 powf 每 token），现在全局算一次。
static float g_inv_swa[256], g_inv_full[64];
static int   g_inv_ready = 0;
static void rope_init_inv(void) {
    if (g_inv_ready) return;
    for (int i = 0; i < 128; i++) g_inv_swa[i]  = powf(10000.0f, -2.0f * (float)i / 256.0f);
    for (int i = 0; i < 32;  i++) g_inv_full[i] = powf(5000000.0f, -2.0f * (float)i / 64.0f);
    g_inv_ready = 1;
}

static void rope_apply(float *v, int n_heads, int head_dim, int half,
                       const float *cs, const float *sn) {
    for (int h = 0; h < n_heads; h++) {
        float *p = v + (size_t)h * head_dim;
        int i = 0;
        for (; i + 4 <= half; i += 4) {
            const float32x4_t c = vld1q_f32(cs + i), sn4 = vld1q_f32(sn + i);
            const float32x4_t a = vld1q_f32(p + i), b = vld1q_f32(p + i + half);
            vst1q_f32(p + i,      vsubq_f32(vmulq_f32(a, c), vmulq_f32(b, sn4)));
            vst1q_f32(p + i + half, vaddq_f32(vmulq_f32(a, sn4), vmulq_f32(b, c)));
        }
        for (; i < half; i++) {
            const float a = p[i], b = p[i + half];
            p[i]        = a * cs[i] - b * sn[i];
            p[i + half] = a * sn[i] + b * cs[i];
        }
    }
}

// q 和 k 共用一组 cos/sin：两者 rdim/base/pos 相同，原来各算一遍 cosf/sinf
static void rope_pair(float *q, int nqh, float *k, int nkh, int head_dim,
                      int rope_dims, float base, int pos) {
    const int half = rope_dims / 2;
    if (half <= 0 || half > 512) return;
    rope_init_inv();
    const float *inv_tab = (rope_dims == 64) ? g_inv_full : g_inv_swa;
    (void)base;
    float cs[512], sn[512];
    for (int i = 0; i < half; i++) {
        const float ang = (float)pos * inv_tab[i];
        cs[i] = cosf(ang);
        sn[i] = sinf(ang);
    }
    rope_apply(q, nqh, head_dim, half, cs, sn);
    rope_apply(k, nkh, head_dim, half, cs, sn);
}

static void rope(float *v, int n_heads, int head_dim, int rope_dims,
                 float base, int pos) {
    const int half = rope_dims / 2;
    if (half <= 0 || half > 512) return;
    rope_init_inv();

    const float *inv_tab = (rope_dims == 64) ? g_inv_full : g_inv_swa;
    (void)base;

    float cs[512], sn[512];
    for (int i = 0; i < half; i++) {
        const float ang = (float)pos * inv_tab[i];
        cs[i] = cosf(ang);
        sn[i] = sinf(ang);
    }

    for (int h = 0; h < n_heads; h++) {
        float *p = v + (size_t)h * head_dim;
        // 向量化：一次 4 个频率对。RoPE 是纯逐元素运算，
        // 标量循环在 36 层 × 20 head × 128 对上要 10ms（实测占非 GEMV 的 40%）。
        int i = 0;
        for (; i + 4 <= half; i += 4) {
            const float32x4_t c = vld1q_f32(cs + i), sn4 = vld1q_f32(sn + i);
            const float32x4_t a = vld1q_f32(p + i), b = vld1q_f32(p + i + half);
            vst1q_f32(p + i,      vsubq_f32(vmulq_f32(a, c), vmulq_f32(b, sn4)));
            vst1q_f32(p + i + half, vaddq_f32(vmulq_f32(a, sn4), vmulq_f32(b, c)));
        }
        for (; i < half; i++) {
            const float a = p[i], b = p[i + half];
            p[i] = a * cs[i] - b * sn[i];
            p[i + half] = a * sn[i] + b * cs[i];
        }
    }
}

// ── 引擎生命周期 ─────────────────────────────────────────
Engine *engine_open(const char *path, int max_ctx) {
    // 【3 线程，已定论】15 轮取中位、四分位很窄的可靠数据：
    //   T=1 2.769 / T=2 4.247 / T=3 4.597 / T=4 4.252
    // T=3 比 T=4 快 8%。不再来回切 —— 之前那些波动是单次测量的噪声。
    return engine_open_t(path, max_ctx, 3);
}

Engine *engine_open_t(const char *path, int max_ctx, int nthreads) {
    GgufModel *gm = gguf_open(path);
    if (!gm) return NULL;
    if (nthreads > 1) {
        pool_init(nthreads);
        gemv_set_threads(pool_threads());
    } else {
        gemv_set_threads(1);
    }

    Engine *e = calloc(1, sizeof(Engine));
    e->gm = gm;
    e->H  = (int)gm->n_embd;
    e->NH = (int)gm->n_head;
    e->NKV= (int)gm->n_head_kv;
    e->HD = gm->head_dim;
    e->FF = (int)gm->n_ff;
    e->L  = (int)gm->n_layer;
    e->V  = (int)gm->vocab_size;
    e->max_ctx = max_ctx;
    e->rope_dims_swa  = (int)gm->rope_dims;      // 64? 用元数据，S 层单独处理
    e->rope_base_full = gm->rope_theta;

    const int qd = e->NH * e->HD, kvd = e->NKV * e->HD;
    e->x    = calloc(e->H, sizeof(float));
    e->h    = calloc(e->H, sizeof(float));
    e->qkv  = calloc(qd + 2 * kvd, sizeof(float));
    e->attn = calloc(qd, sizeof(float));
    e->gate = calloc(e->NH, sizeof(float));
    e->ff1  = calloc(e->FF, sizeof(float));
    e->ff2  = calloc(e->FF, sizeof(float));
    e->logits = calloc(e->V, sizeof(float));

    // 激活量化的临时缓冲（取 H 和 FF 的较大者）
    const int nA = e->H > e->FF ? e->H : e->FF;
    // posix_memalign 而不是 aligned_alloc —— 后者在 Termux 的 clang 21 上
    // 需要 C11 的 <stdlib.h> 声明，编译会报 undeclared
    if (posix_memalign((void **)&e->qa, 64, (size_t)nA + 64) ||
        posix_memalign((void **)&e->bsa, 64, ((size_t)nA / 16 + 8) * 2) ||
        posix_memalign((void **)&e->asc, 64, ((size_t)nA / 32 + 8) * sizeof(float)) ||
        posix_memalign((void **)&e->qa8, 64, (size_t)nA + 64) ||
        posix_memalign((void **)&e->asc8, 64, ((size_t)nA / 256 + 8) * sizeof(float)) ||
        posix_memalign((void **)&e->bsa8, 64, ((size_t)nA / 16 + 8) * 2)) {
        engine_close(e);
        return NULL;
    }

    // KV cache：每层 k/v 各 [max_ctx][NKV][HD]，存 fp16 省一半内存
    e->kcache = calloc(e->L, sizeof(__fp16 *));
    e->vcache = calloc(e->L, sizeof(__fp16 *));
    const size_t per = (size_t)max_ctx * kvd;
    for (int i = 0; i < e->L; i++) {
        e->kcache[i] = calloc(per, sizeof(__fp16));
        e->vcache[i] = calloc(per, sizeof(__fp16));
    }
    e->pos = 0;
    // 重排最热的 Q4_K 张量（ffn_gate/ffn_up）供 8 路内核使用
    x8_init(gm, nthreads);
    // int8 预解包默认关闭：权重从 210 字节涨到 256 字节（+22%），
    // 在小张量上热读能快 1.95 倍，但真实冷读时多读的 22% 是净亏。
    if (getenv("I8_ENABLE")) i8_init(gm);
    // 【预解析张量指针】每层的固定张量一次性查找好
    {
        const int L = e->L;
        e->t_qkv       = calloc(L, sizeof(void*));
        e->t_attn_out  = calloc(L, sizeof(void*));
        e->t_ffn_gate  = calloc(L, sizeof(void*));
        e->t_ffn_up    = calloc(L, sizeof(void*));
        e->t_ffn_down  = calloc(L, sizeof(void*));
        e->t_attn_gate = calloc(L, sizeof(void*));
        char nm[96];
        #define LOOK(arr, fmt) do{ snprintf(nm,sizeof nm,fmt,l); \
            e->arr[l] = gguf_find(gm, nm); }while(0)
        for (int l = 0; l < L; l++) {
            LOOK(t_qkv,       "blk.%d.attn_qkv.weight");
            LOOK(t_attn_out,  "blk.%d.attn_output.weight");
            LOOK(t_ffn_gate,  "blk.%d.ffn_gate.weight");
            LOOK(t_ffn_up,    "blk.%d.ffn_up.weight");
            LOOK(t_ffn_down,  "blk.%d.ffn_down.weight");
            LOOK(t_attn_gate, "blk.%d.attn_gate.weight");
        }
        #undef LOOK
        e->t_emb = gguf_find(gm, "token_embd.weight");
    }

    // 【预转换 norm 权重】只做一次，省掉每层 3 次 malloc + tensor_to_f32
    {
        const int H = e->H, L = e->L;
        e->norm_attn = malloc(sizeof(float) * H * L);
        e->norm_ffn  = malloc(sizeof(float) * H * L);
        e->norm_out  = malloc(sizeof(float) * H);
        char nm[96];
        for (int l = 0; l < L; l++) {
            snprintf(nm, sizeof nm, "blk.%d.attn_norm.weight", l);
            GgufTensor *t = gguf_find(gm, nm);
            if (t) tensor_to_f32(t, e->norm_attn + (size_t)l * H, H);
            snprintf(nm, sizeof nm, "blk.%d.ffn_norm.weight", l);
            t = gguf_find(gm, nm);
            if (t) tensor_to_f32(t, e->norm_ffn + (size_t)l * H, H);
        }
        GgufTensor *on = gguf_find(gm, "output_norm.weight");
        if (on) tensor_to_f32(on, e->norm_out, H);
    }

    return e;
}

void engine_close(Engine *e) {
    if (!e) return;
    for (int i = 0; i < e->L; i++) { free(e->kcache[i]); free(e->vcache[i]); }
    free(e->kcache); free(e->vcache);
    free(e->x); free(e->h); free(e->qkv); free(e->attn); free(e->gate);
    free(e->ff1); free(e->ff2); free(e->logits); free(e->qa); free(e->bsa); free(e->asc);
    free(e->qa8); free(e->asc8); free(e->bsa8);
    free(e->norm_attn); free(e->norm_ffn); free(e->norm_out);
    free(e->t_qkv); free(e->t_attn_out); free(e->t_ffn_gate);
    free(e->t_ffn_up); free(e->t_ffn_down); free(e->t_attn_gate);
    gguf_close(e->gm);
    free(e);
}

// ── 单层 ─────────────────────────────────────────────────
// 单个位置的前向。pos 显式传入，便于批量验证时每个位置独立。
// x 是输入/输出的隐藏状态（就地更新）。
static void layer_forward_at(Engine *e, int L, float *x, int pos) {
    GgufModel *gm = e->gm;
    const int H = e->H, NH = e->NH, NKV = e->NKV, HD = e->HD;
    const int qd = NH * HD, kvd = NKV * HD;
    char name[64];

    // 层类型：[S,S,S,F] 循环。用元数据的 sliding_window_pattern 判断不了，
    // 官方 config 是每 4 层一个 full（第 3、7、11... 层）
    const int is_full = ((L + 1) % 4) == 0;

    // ── 注意力 ──
    TSTA();
    rmsnorm(x, e->norm_attn + (size_t)L * H, e->h, H, (float)gm->rms_eps);
    TEND(TK_NORM);

    if (getenv("DBG_LAYER") && L == 0) {
        double h2 = 0; for (int i = 0; i < H; i++) h2 += (double)e->h[i]*e->h[i];
        fprintf(stderr, "  S L0 rmsnorm后 |h|=%.4f h[0..2]=%.4f %.4f\n", sqrt(h2), e->h[0], e->h[1]);
    }
    // QKV
    GgufTensor *qkvw = e->t_qkv[L];
    // per-block 量化：scale 已经在内核里按块乘掉了，不再需要整体乘回
    TSTA();
    quantize_act_b(e->h, e->qa, e->asc, e->bsa, H);
    TEND(TK_QUANT);
    // attn_qkv 走 int8 缓存收益为负：每层要多做一次 per-256 量化，
    // 而它只有 12.3MB，省下的时间抵不过量化开销。保持原路径。
    gemv_rows_f(qkvw, e->h, e->qa, e->asc, e->bsa, e->qkv, 0, (size_t)(qd + 2 * kvd));

    float *q = e->qkv, *k = e->qkv + qd, *v = e->qkv + qd + kvd;

    // partial RoPE
    const int rdim = is_full ? 64 : HD;
    const float rbase = is_full ? e->rope_base_full : 10000.0f;
    // q 和 k 用同一组 cos/sin，原来各算一遍（每层 128 个 cosf+sinf 重复）
    TSTA();
    rope_pair(q, NH, k, NKV, HD, rdim, rbase, pos);
    TEND(TK_ROPE);

    // 写入 KV cache
    const size_t off = (size_t)pos * kvd;
    for (int i = 0; i < kvd; i++) {
        e->kcache[L][off + i] = (__fp16)k[i];
        e->vcache[L][off + i] = (__fp16)v[i];
    }

        // 注意力：窗口内的位置
    const int win = is_full ? pos + 1
                            : ((int)gm->window > 0 && pos + 1 > (int)gm->window
                                   ? (int)gm->window : pos + 1);
    const int t0 = pos + 1 - win;
    const float scale = 1.0f / sqrtf((float)HD);

    TSTA();
    for (int h = 0; h < NH; h++) {
        const int kvh = h / (NH / NKV);       // GQA
        float *qh = q + (size_t)h * HD;
        float *out = e->attn + (size_t)h * HD;

        // 【向量化 + 一遍 softmax】原来每个 head 要扫两遍 KV、全是标量乘加，
        // 每步 512x128x2 = 131K 次乘加 x 16 head = 2.1M，非常慢。
        // 现在：
        //   1. QK 点积用 vfma 累加，fp16 K 用 vcvt 一次转 8 个
        //   2. 用在线 softmax（先累加未归一化的 exp 和加权 V，最后统一除）
        //      省掉第一遍找 max —— decode 时 q·k/sqrt(d) 的量级稳定在 ±30 内，
        //      直接 exp 不会溢出（exp(30) = 1e13，float 够）
        float32x4_t accv[64];
        const int nvec = HD / 4;
        for (int i = 0; i < nvec; i++) accv[i] = vdupq_n_f32(0.f);
        float sum = 0.f;

        for (int t = t0; t <= pos; t++) {
            const __fp16 *kt = e->kcache[L] + (size_t)t * kvd + (size_t)kvh * HD;
            // QK 点积，8 个 fp16 一组
            float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
            for (int d = 0; d < HD; d += 8) {
                const float16x8_t kv = vld1q_f16(kt + d);
                a0 = vfmaq_f32(a0, vld1q_f32(qh + d),     vcvt_f32_f16(vget_low_f16(kv)));
                a1 = vfmaq_f32(a1, vld1q_f32(qh + d + 4), vcvt_high_f32_f16(kv));
            }
            const float sc = vaddvq_f32(vaddq_f32(a0, a1)) * scale;
            const float p = expf(sc);
            sum += p;

            const __fp16 *vt2 = e->vcache[L] + (size_t)t * kvd + (size_t)kvh * HD;
            const float32x4_t pv = vdupq_n_f32(p);
            for (int i = 0; i < nvec; i++) {
                const float16x8_t vv = vld1q_f16(vt2 + i * 4);
                // 只取前 4 个（nvec 是按 4 分组的）
                accv[i] = vfmaq_f32(accv[i], pv, vcvt_f32_f16(vget_low_f16(vv)));
            }
        }
        const float inv = (sum > 1e-30f) ? 1.0f / sum : 0.f;
        for (int i = 0; i < nvec; i++) vst1q_f32(out + i * 4, vmulq_n_f32(accv[i], inv));
    }
    TEND(TK_ATTN);


    // headwise sigmoid gate
    GgufTensor *gw = e->t_attn_gate[L];
    if (gw) {
        // per-block 版：scale 已在核心里乘掉
        gemv_rows_b(gw, e->qa, e->asc, NULL, e->gate, 0, (size_t)NH);
        for (int h = 0; h < NH; h++) {
            const float g = 1.0f / (1.0f + expf(-e->gate[h]));
            float *o = e->attn + (size_t)h * HD;
            for (int d = 0; d < HD; d++) o[d] *= g;
        }
    }

    // output proj + 残差
    GgufTensor *ow = e->t_attn_out[L];
    // attn_output 之前用 per-tensor 量化（gemv_rows + 外部乘 so），
    // 实测效率只有 68%，而 per-block 版的其他张量都在 93%+。
    // 改成 per-block 后能到同样的水平。
    quantize_act_b(e->attn, e->qa, e->asc, e->bsa, qd);
    gemv_rows_f(ow, e->attn, e->qa, e->asc, e->bsa, e->h, 0, (size_t)H);
    for (int i = 0; i < H; i++) x[i] += e->h[i];

    if (getenv("DBG_LAYER") && L == 0) {
        double n2 = 0; for (int i = 0; i < H; i++) n2 += (double)x[i]*x[i];
        fprintf(stderr, "S%02d attn后 |x|=%.4f  x[0..2]=%.4f %.4f %.4f\n", L, sqrt(n2), x[0], x[1], x[2]);
    }
    // ── MLP（GeGLU）──
    // norm 权重已预转换成 float 缓存
    rmsnorm(x, e->norm_ffn + (size_t)L * H, e->h, H, (float)gm->rms_eps);

    GgufTensor *fg = e->t_ffn_gate[L];
    GgufTensor *fu = e->t_ffn_up[L];
    GgufTensor *fd = e->t_ffn_down[L];

    quantize_act_b(e->h, e->qa, e->asc, e->bsa, H);
    // gate 和 up 输入相同（都是 e->h），原来各调一次 x8_prepare
    // 把 2560 个元素量化两遍。这里量化一次、两个 GEMV 共用。
    if (fg->x8_type || fu->x8_type) {
        quant_act_q8k(e->h, e->qa8, e->asc8, e->bsa8, H);
        gemv_rows_pre(fg, e->qa8, e->asc8, e->bsa8, e->qa, e->asc, e->bsa,
                      e->ff1, 0, (size_t)e->FF);
        gemv_rows_pre(fu, e->qa8, e->asc8, e->bsa8, e->qa, e->asc, e->bsa,
                      e->ff2, 0, (size_t)e->FF);
    } else {
        gemv_rows_f(fg, e->h, e->qa, e->asc, e->bsa, e->ff1, 0, (size_t)e->FF);
        gemv_rows_f(fu, e->h, e->qa, e->asc, e->bsa, e->ff2, 0, (size_t)e->FF);
    }

    // gelu(gate) * up（per-block 版已经在核心里乘过 scale，这里直接用）
    //
    // 【向量化】tanhf 在 ARM 上是库函数，每次约 20 cycle，10240 个就是
    // 20 万 cycle ≈ 78us/层，36 层 2.8ms —— 实测 GeGLU 占非 GEMV 开销的 25%。
    // 改用 GELU 的有理近似（误差 < 1e-4，对 FP16 精度绰绰有余）：
    //   gelu(x) ≈ x * sigmoid(1.702x)  的 tanh 形式：
    //   t = tanh(0.7978845608*(x + 0.044715x³))
    //   tanh 用 Padé 近似：tanh(z) ≈ z*(27+z²)/(27+9z²)，|z|<3 时误差 <2e-3
    //   |z|>=3 时 tanh 已经饱和到 ±1，直接夹住。
    {
        const float32x4_t k1 = vdupq_n_f32(0.7978845608f);
        const float32x4_t k2 = vdupq_n_f32(0.044715f);
        const float32x4_t half = vdupq_n_f32(0.5f);
        const float32x4_t one = vdupq_n_f32(1.0f);
        const float32x4_t c27 = vdupq_n_f32(27.0f);
        const float32x4_t c9  = vdupq_n_f32(9.0f);
        const float32x4_t lim = vdupq_n_f32(3.0f);
        const float32x4_t nlim= vdupq_n_f32(-3.0f);
        const float32x4_t sgn = vdupq_n_f32(1.0f);
        const float32x4_t msgn= vdupq_n_f32(-1.0f);

        TSTA();
        int i = 0;
        for (; i + 4 <= e->FF; i += 4) {
            const float32x4_t g = vld1q_f32(e->ff1 + i);
            const float32x4_t u = vld1q_f32(e->ff2 + i);
            const float32x4_t z = vmulq_f32(k1, vfmaq_f32(g, k2, vmulq_f32(g, vmulq_f32(g, g))));
            // Padé
            const float32x4_t z2 = vmulq_f32(z, z);
            const float32x4_t num = vmulq_f32(z, vfmaq_f32(c27, z2, one));
            const float32x4_t den = vfmaq_f32(c27, z2, c9);
            float32x4_t t = vdivq_f32(num, den);
            // 饱和区夹到 ±1
            t = vbslq_f32(vcgtq_f32(z, lim), sgn, t);
            t = vbslq_f32(vcltq_f32(z, nlim), msgn, t);
            // 0.5*g*(1+t)*u
            const float32x4_t r = vmulq_f32(vmulq_f32(vmulq_f32(half, g), vaddq_f32(one, t)), u);
            vst1q_f32(e->ff1 + i, r);
        }
        for (; i < e->FF; i++) {
            const float g = e->ff1[i];
            const float u = e->ff2[i];
            const float t = tanhf(0.7978845608f * (g + 0.044715f * g * g * g));
            e->ff1[i] = 0.5f * g * (1.0f + t) * u;
        }
        TEND(TK_GEGLU);
    }

    // ffn_down 的输入是 10240 维（FF），per-block 量化的开销比 per-tensor
    // 大得多（要多算 320 组的 scale）。实测改成 per-block 后反而从 4.11 掉到 3.90，
    // 所以保留 per-tensor。
    // ffn_down 走 x8（在 kHot4 清单里）：输入 10240 维，x8_range_f 内部
    // 按 per-256 量化。非 x8 张量退回 per-tensor（要乘 sd 反量化）。
    {
        const int is_x8 = fd->x8_type != 0;
        const float sd = is_x8 ? 1.0f : quantize_act(e->ff1, e->qa, e->bsa, e->FF);
        if (is_x8) gemv_rows_f(fd, e->ff1, e->qa, e->asc, e->bsa, e->h, 0, (size_t)H);
        else       gemv_rows(fd, e->qa, e->bsa, e->h, 0, (size_t)H);
        for (int i = 0; i < H; i++) x[i] += e->h[i] * sd;
    }
}

// ── 单 token 前向 ────────────────────────────────────────
float *engine_step(Engine *e, int token) {
    GgufModel *gm = e->gm;
    const int H = e->H;

    // embedding（token_embd 与 lm_head 绑定）
    GgufTensor *emb = e->t_emb;
    if (!emb) return NULL;
    size_t bs;
    switch (emb->type) {
    case 10: bs = 84;  break;
    case 11: bs = 110; break;
    case 12: bs = 144; break;
    case 14: bs = 210; break;
    default: bs = 144;
    }
    const size_t rb = H / 256 * bs;
    {
        // 直接解压到 e->x，省掉每 token 一次 malloc + memcpy（128KB）
        extern void dequant_q2k_row(const void *, float *, size_t);
        extern void dequant_q3k_row(const void *, float *, size_t);
        extern void dequant_q4k_row(const void *, float *, size_t);
        extern void dequant_q6k_row(const void *, float *, size_t);
        const void *row = emb->data + (size_t)token * rb;
        switch (emb->type) {
        case 10: dequant_q2k_row(row, e->x, H); break;
        case 11: dequant_q3k_row(row, e->x, H); break;
        case 14: dequant_q6k_row(row, e->x, H); break;
        default: dequant_q4k_row(row, e->x, H); break;
        }
    }
    if (getenv("DBG_LAYER")) {
        double n2 = 0; for (int i = 0; i < H; i++) n2 += (double)e->x[i]*e->x[i];
        fprintf(stderr, "S-- emb |x|=%.4f  x[0..2]=%.4f %.4f %.4f\n", sqrt(n2), e->x[0], e->x[1], e->x[2]);
    }

    for (int L = 0; L < e->L; L++) {
        layer_forward_at(e, L, e->x, e->pos);
        if (getenv("DBG_CMP")) {
            double n2 = 0; for (int i = 0; i < H; i++) n2 += (double)e->x[i]*e->x[i];
            fprintf(stderr, "S%02d last|x|=%.5f x0=%.5f\n", L, sqrt(n2), e->x[0]);
        }
        if (getenv("DBG_LAYER")) {
            double n2 = 0; for (int i = 0; i < H; i++) n2 += (double)e->x[i]*e->x[i];
            fprintf(stderr, "S%02d |x|=%.4f  x[0..2]=%.4f %.4f %.4f\n",
                    L, sqrt(n2), e->x[0], e->x[1], e->x[2]);
        }
    }

    // 最终 norm + lm_head
    if (e->norm_out)
        rmsnorm(e->x, e->norm_out, e->h, H, (float)gm->rms_eps);

    // lm_head：131072 x 2560，用 tied embedding
    quantize_act_b(e->h, e->qa, e->asc, e->bsa, H);
    gemv_rows_b(emb, e->qa, e->asc, e->bsa, e->logits, 0, (size_t)e->V);

    e->pos++;
    return e->logits;
}

// ── 批量前向（投机解码验证用）───────────────────────────
// 一次前向同时处理 n 个位置，KV cache 按 [pos0 .. pos0+n) 写入。
// 这是投机解码的核心：验证 k 个 draft token 时，target 只跑一次前向，
// 而不是 k 次 —— 否则验证成本就是 k 倍，收益全被吃掉。
//
// 【为什么不能简单地循环调用 layer_forward_at】
// 那会让每个位置的权重被解 k 遍。decode 本来就卡在解量化的 ALU 上，
// k 倍开销直接把投机解码的收益吃光。
// 所以这里用 gemm_rows：权重解一次，和 n 份激活各做一次 SDOT。

typedef struct {
    float *x;        // [n][H] 隐藏状态
    int8_t *qa;      // [n][H] 激活量化
    int16_t *bsa;    // [n][H/16+1]
    float *tmp;      // [n][max(H,FF)]
} BatchBuf;

// 批量 rmsnorm
static void rmsnorm_n(const float *x, const float *w, float *out, int n, int H, float eps) {
    for (int t = 0; t < n; t++)
        rmsnorm(x + (size_t)t * H, w, out + (size_t)t * H, H, eps);
}

// 批量量化激活。qstride/bstride 是每个 token 的步长（可大于 H，留出余量）
static void quantize_n(const float *x, int8_t *q, int16_t *bs, int n, int H,
                       size_t qstride, size_t bstride) {
    for (int t = 0; t < n; t++)
        quantize_act(x + (size_t)t * H, q + (size_t)t * qstride,
                     bs ? bs + (size_t)t * bstride : NULL, H);
}

// 批量量化（per-32）：给 Q4_K 张量用，和单 token 路径的 quantize_act_b 一致。
// 和 per-tensor 并存 —— attn_qkv 是 Q6_K，q6k_gemm_row 不支持 per-32。
static void quantize_n32(const float *x, int8_t *q, int16_t *bs, float *asc,
                         int n, int H, size_t qstride, size_t bstride, size_t astride) {
    extern void quant_act_block(const float *, int8_t *, float *, int16_t *, int);
    for (int t = 0; t < n; t++)
        quant_act_block(x + (size_t)t * H, q + (size_t)t * qstride,
                        asc ? asc + (size_t)t * astride : NULL,
                        bs ? bs + (size_t)t * bstride : NULL, H);
}


// 批量 GEMM 的辅助：qa/bsa 的 stride 是 maxdim，而 gemm_rows 要求紧密 [n][cols]。
// 这里打包一份紧密的再调用，代价是 n*cols 字节的拷贝（比权重小三个数量级）。
extern void gemm_rows_a(GgufTensor *t, const int8_t *X, const int16_t *bs,
                        const float *asc, float *Y, int n, size_t rows);

// asc 非空时按 per-32 激活 scale，结果已是最终值（调用方不要再乘 scale）
static void gemm_packed_asc(GgufTensor *t, const int8_t *qa, const int16_t *bsa,
                            const float *asc, size_t qa_stride, size_t bs_stride,
                            size_t asc_stride, float *Y, int n, size_t cols, size_t rows) {
    int8_t *xp = malloc((size_t)n * cols);
    int16_t *bp = malloc((size_t)n * (cols / 16 + 1) * 2);
    float *ap = malloc((size_t)n * (cols / 32 + 1) * sizeof(float));
    for (int t2 = 0; t2 < n; t2++) {
        memcpy(xp + (size_t)t2 * cols, qa + (size_t)t2 * qa_stride, cols);
        if (bsa) memcpy(bp + (size_t)t2 * (cols / 16 + 1),
                        (const char *)bsa + (size_t)t2 * bs_stride * 2,
                        (cols / 16 + 1) * 2);
        if (asc) memcpy(ap + (size_t)t2 * (cols / 32 + 1),
                        asc + (size_t)t2 * asc_stride, (cols / 32 + 1) * sizeof(float));
    }
    gemm_rows_a(t, xp, bsa ? bp : NULL, asc ? ap : NULL, Y, n, rows);
    free(xp); free(bp); free(ap);
}

static void gemm_packed(GgufTensor *t, const int8_t *qa, const int16_t *bsa,
                        size_t qa_stride, size_t bs_stride,
                        float *Y, int n, size_t cols, size_t rows) {
    int8_t *xp = malloc((size_t)n * cols);
    int16_t *bp = malloc((size_t)n * (cols / 16 + 1) * 2);
    for (int t2 = 0; t2 < n; t2++) {
        memcpy(xp + (size_t)t2 * cols, qa + (size_t)t2 * qa_stride, cols);
        if (bsa) memcpy(bp + (size_t)t2 * (cols / 16 + 1),
                        (const char *)bsa + (size_t)t2 * bs_stride * 2,
                        (cols / 16 + 1) * 2);
    }
    gemm_rows(t, xp, bsa ? bp : NULL, Y, n, rows);
    free(xp); free(bp);
}

float *engine_step_batch(Engine *e, const int *tokens, int n) {
    if (n <= 0) return NULL;
    if (n == 1 && getenv("NO_BATCH_SHORTCUT") == NULL) return engine_step(e, tokens[0]);

    const int H = e->H, FF = e->FF;
    const int NH = e->NH, NKV = e->NKV, HD = e->HD;
    const int qd = NH * HD, kvd = NKV * HD;
    const int pos0 = e->pos;
    GgufModel *gm = e->gm;
    char name[64];

    // 缓冲：n 份隐藏状态 + 激活 + 中间结果
    // 【stride 必须统一】qa/bsa 在不同阶段装 H 维或 FF 维的激活，
    // 如果按各自维度算 stride，切换阶段时就会错位。统一用 maxdim。
    // stride 要比 maxdim 多留 64 字节的余量：quantize_act 在算 bsums 时
    // 会按 16 个一组向上取整，最后不足 16 个的部分也可能写满一格。
    const int maxdim = H > FF ? H : FF;
    const size_t qa_stride = (size_t)maxdim + 64;
    const size_t bs_stride = (size_t)maxdim / 16 + 8;
    const size_t asc_stride = (size_t)maxdim / 32 + 8;
    float *x    = calloc((size_t)n * H, sizeof(float));
    int8_t *qa  = calloc((size_t)n * qa_stride, 1);
    int16_t *bsa= calloc((size_t)n * bs_stride, 2);
    float *qkv  = calloc((size_t)n * (qd + 2 * kvd), sizeof(float));
    // per-32 专用缓冲（和 qa/bsa 的 per-tensor 并存）
    float   *asc   = calloc((size_t)n * asc_stride, sizeof(float));
    int8_t  *qa32  = calloc((size_t)n * qa_stride, 1);
    int16_t *bsa32 = calloc((size_t)n * bs_stride, 2);

    // embedding 逐位置填
    GgufTensor *emb = e->t_emb;
    if (!emb) { free(x); free(qa); free(bsa); free(qkv); return NULL; }
    {
        size_t bsb;
        switch (emb->type) {
        case 10: bsb = 84;  break;
        case 11: bsb = 110; break;
        case 12: bsb = 144; break;
        default: bsb = 210; break;
        }
        const size_t rb = H / 256 * bsb;
        extern void dequant_q2k_row(const void *, float *, size_t);
        extern void dequant_q3k_row(const void *, float *, size_t);
        extern void dequant_q4k_row(const void *, float *, size_t);
        extern void dequant_q6k_row(const void *, float *, size_t);
        for (int t = 0; t < n; t++) {
            const void *row = emb->data + (size_t)tokens[t] * rb;
            switch (emb->type) {
            case 10: dequant_q2k_row(row, x + (size_t)t * H, H); break;
            case 11: dequant_q3k_row(row, x + (size_t)t * H, H); break;
            case 14: dequant_q6k_row(row, x + (size_t)t * H, H); break;
            default: dequant_q4k_row(row, x + (size_t)t * H, H); break;
            }
        }
    }

    if (getenv("DBG_LAYER")) {
        double n2 = 0; for (int i = 0; i < H; i++) n2 += (double)x[i]*x[i];
        fprintf(stderr, "B-- emb |x|=%.4f  x[0..2]=%.4f %.4f %.4f\n", sqrt(n2), x[0], x[1], x[2]);
    }
        if (getenv("DBG_STAGE")) fprintf(stderr, "[stage] emb done\n");
    float *h = calloc((size_t)n * H, sizeof(float));

    for (int L = 0; L < e->L; L++) {
        const int is_full = ((L + 1) % 4) == 0;

        snprintf(name, sizeof name, "blk.%d.attn_norm.weight", L);
        GgufTensor *norm1 = gguf_find(gm, name);
        if (!norm1) break;
        float *nw = malloc(H * sizeof(float));
        tensor_to_f32(norm1, nw, H);
        rmsnorm_n(x, nw, h, n, H, (float)gm->rms_eps);
        free(nw);

        // QKV 批量
        snprintf(name, sizeof name, "blk.%d.attn_qkv.weight", L);
        GgufTensor *qkvw = gguf_find(gm, name);
        quantize_n(h, qa, bsa, n, H, qa_stride, bs_stride);
        {
            // h 已经由 quantize_n 量化进 qa，这里只需要拿回 scale
            // 【必须用 per-32】qkv 的误差会直接进 KV cache，再被 36 层
            // attention 指数放大 —— 用 per-tensor 时位置 0 的 L00 就差 0.0024，
            // 到输出层放大成 22.9。attn_qkv 是 Q6_K，q6k_gemm_row_a 现在支持 asc。
            const int qkv32 = (qkvw && (qkvw->type == 12 || qkvw->type == 14));
            float *Y = calloc((size_t)n * (qd + 2 * kvd), sizeof(float));
            if (qkv32) {
                quantize_n32(h, qa32, bsa32, asc, n, H, qa_stride, bs_stride, asc_stride);
                gemm_packed_asc(qkvw, qa32, bsa32, asc, qa_stride, bs_stride, asc_stride,
                                Y, n, (size_t)H, (size_t)(qd + 2 * kvd));
            } else {
                quantize_n(h, qa, bsa, n, H, qa_stride, bs_stride);
                gemm_packed(qkvw, qa, bsa, qa_stride, bs_stride,
                            Y, n, (size_t)H, (size_t)(qd + 2 * kvd));
                float *scales = calloc(n, sizeof(float));
                for (int t = 0; t < n; t++) {
                    float mx = 0.f;
                    for (int i = 0; i < H; i++) { float a = fabsf(h[(size_t)t*H+i]); if (a > mx) mx = a; }
                    scales[t] = (mx > 1e-8f) ? mx / 127.0f : 1.0f;
                    for (int i = 0; i < qd + 2 * kvd; i++)
                        Y[(size_t)t*(qd+2*kvd)+i] *= scales[t];
                }
                free(scales);
            }
            for (int t = 0; t < n; t++)
                for (int i = 0; i < qd + 2 * kvd; i++)
                    qkv[(size_t)t * (qd + 2 * kvd) + i] =
                        Y[(size_t)t * (qd + 2 * kvd) + i];
            free(Y);
        }

        if (getenv("DBG_STAGE")) fprintf(stderr, "[stage] L%d qkv后\n", L);
        // 逐位置做 RoPE + 写 KV + 注意力
        float *attnout = calloc((size_t)n * qd, sizeof(float));
        for (int t = 0; t < n; t++) {
            float *q = qkv + (size_t)t * (qd + 2 * kvd);
            float *k = q + qd;
            float *v = q + qd + kvd;
            const int pos = pos0 + t;

            const int rdim = is_full ? 64 : HD;
            const float rbase = is_full ? e->rope_base_full : 10000.0f;
            rope(q, NH, HD, rdim, rbase, pos);
            rope(k, NKV, HD, rdim, rbase, pos);

            const size_t off = (size_t)pos * kvd;
            for (int i = 0; i < kvd; i++) {
                e->kcache[L][off + i] = (__fp16)k[i];
                e->vcache[L][off + i] = (__fp16)v[i];
            }

            const int win = is_full ? pos + 1
                : ((int)gm->window > 0 && pos + 1 > (int)gm->window ? (int)gm->window : pos + 1);
            const int t0 = pos + 1 - win;
            const float scale = 1.0f / sqrtf((float)HD);

            for (int hh = 0; hh < NH; hh++) {
                const int kvh = hh / (NH / NKV);
                float *qh = q + (size_t)hh * HD;
                float *o  = attnout + (size_t)t * qd + (size_t)hh * HD;
                float mx = -INFINITY;
                for (int p2 = t0; p2 <= pos; p2++) {
                    const __fp16 *kt = e->kcache[L] + (size_t)p2 * kvd + (size_t)kvh * HD;
                    float s = 0.f;
                    for (int d = 0; d < HD; d++) s += qh[d] * (float)kt[d];
                    s *= scale;
                    if (s > mx) mx = s;
                }
                float sum = 0.f;
                for (int p2 = t0; p2 <= pos; p2++) {
                    const __fp16 *kt = e->kcache[L] + (size_t)p2 * kvd + (size_t)kvh * HD;
                    float s = 0.f;
                    for (int d = 0; d < HD; d++) s += qh[d] * (float)kt[d];
                    const float p = expf(s * scale - mx);
                    sum += p;
                    const __fp16 *vt = e->vcache[L] + (size_t)p2 * kvd + (size_t)kvh * HD;
                    for (int d = 0; d < HD; d++) o[d] += p * (float)vt[d];
                }
                // sum 理论上恒 >= 1（至少有一个 exp(0)=1），
                // 但如果 mx 是 -inf（窗口为空）就会出 0 除，直接跳过
                if (sum > 1e-30f) {
                    const float inv = 1.0f / sum;
                    for (int d = 0; d < HD; d++) o[d] *= inv;
                }
                if (getenv("DBG_CMP") && L == 3 && t == 1 && hh == 0) {
                    fprintf(stderr, "  B L3 t=1 h0: mx=%.4f sum=%.4f t0=%d pos=%d\n", mx, sum, t0, pos);
                }


            }
        }

        if (getenv("DBG_LAYER") && L == 0) {
            double a2 = 0, q2 = 0;
            for (int i = 0; i < qd; i++) a2 += (double)attnout[i]*attnout[i];
            for (int i = 0; i < qd + 2*kvd; i++) q2 += (double)qkv[i]*qkv[i];
            fprintf(stderr, "  L0 |qkv|=%.4f |attnout|=%.4f  qkv[0..2]=%.4f %.4f\n",
                    sqrt(q2), sqrt(a2), qkv[0], qkv[1]);
        }
        if (getenv("DBG_STAGE")) fprintf(stderr, "[stage] L%d gate前\n", L);
        if (getenv("DBG_CMP") && L == 3) {
            double a2 = 0; for (int i = 0; i < qd; i++) a2 += (double)attnout[i]*attnout[i];
            fprintf(stderr, "  B L3 gate前 |attnout|=%.4f  a[0..2]=%.4f %.4f\n", sqrt(a2), attnout[0], attnout[1]);
        }
        // headwise sigmoid gate：整层算一次，NH 个输出
        // 用 gemv_rows 走统一量化路径 —— target 的 gate 是 F32、
        // draft 的是 Q2_K，不能只处理 F32/F16。
        snprintf(name, sizeof name, "blk.%d.attn_gate.weight", L);
        GgufTensor *gw = gguf_find(gm, name);
        if (gw) {
            float *gv = calloc(NH, sizeof(float));
            if (getenv("DBG_LAYER") && L == 0) {
                double a2 = 0; for (int i = 0; i < qd; i++) a2 += (double)attnout[i]*attnout[i];
                fprintf(stderr, "  L0 gate前 |attnout|=%.4f\n", sqrt(a2));
            }
            for (int t = 0; t < n; t++) {
                // gemv_rows 输出的是 int 累加和，必须乘回激活的量化 scale，
                // 否则 sigmoid 的输入会大出两个数量级、直接饱和到 0。
                float mxv = 0.f;
                for (int i = 0; i < H; i++) {
                    float a = fabsf(h[(size_t)t * H + i]);
                    if (a > mxv) mxv = a;
                }
                const float gsc = (mxv > 1e-8f) ? mxv / 127.0f : 1.0f;
                // gemv_rows 要求紧密 [H]，而 qa 的 stride 是 qa_stride，
                // 必须拷一份出来（和 lm_head 同样的坑）
                int8_t *gx = malloc(H);
                int16_t *gbs = malloc((H / 16 + 1) * 2);
                memcpy(gx, qa + (size_t)t * qa_stride, H);
                memcpy(gbs, (const char *)bsa + (size_t)t * bs_stride * 2,
                       (H / 16 + 1) * 2);
                gemv_rows(gw, gx, gbs, gv, 0, (size_t)NH);
                free(gx); free(gbs);
                for (int hh = 0; hh < NH; hh++) {
                    const float sg = 1.0f / (1.0f + expf(-gv[hh] * gsc));
                    float *o = attnout + (size_t)t * qd + (size_t)hh * HD;
                    for (int d = 0; d < HD; d++) o[d] *= sg;
                }
            }
            free(gv);
        }

        if (getenv("DBG_CMP") && L == 3) {
            double a2 = 0; for (int i = 0; i < qd; i++) a2 += (double)attnout[i]*attnout[i];
            fprintf(stderr, "  B L3 gate后 |attnout|=%.4f\n", sqrt(a2));
        }
        // output proj 批量
        snprintf(name, sizeof name, "blk.%d.attn_output.weight", L);
        GgufTensor *ow = gguf_find(gm, name);
        {
            // 量化一次，拿到每个位置的 scale；后面反量化用同一个值，
            // 不能重新量化（会把 qa 覆盖掉，导致 GEMM 和 scale 对不上）
            if (getenv("DBG_LAYER") && L == 0) {
                double a2 = 0; for (int i = 0; i < qd; i++) a2 += (double)attnout[i]*attnout[i];
                fprintf(stderr, "  L0 量化前 |attnout|=%.4f  a[0..2]=%.4f %.4f\n",
                        sqrt(a2), attnout[0], attnout[1]);
            }
            // 【按张量类型选量化】per-32 只有 Q4_K 支持 —— q6k_gemm_row 不接收 asc。
            // blk.1 的 attn_output 就是 Q6_K，混用会让结果放大 127 倍。
            const int use32 = (ow && ow->type == 12);
            float *Y = calloc((size_t)n * H, sizeof(float));
            if (use32) {
                quantize_n32(attnout, qa32, bsa32, asc, n, qd, qa_stride, bs_stride, asc_stride);
                gemm_packed_asc(ow, qa32, bsa32, asc, qa_stride, bs_stride, asc_stride, Y, n, (size_t)qd, (size_t)H);
            } else {
                float *scales = calloc(n, sizeof(float));
                for (int t = 0; t < n; t++)
                    scales[t] = quantize_act(attnout + (size_t)t * qd,
                                             qa + (size_t)t * qa_stride,
                                             bsa + (size_t)t * bs_stride, qd);
                gemm_packed(ow, qa, bsa, qa_stride, bs_stride, Y, n, (size_t)qd, (size_t)H);
                for (int t = 0; t < n; t++)
                    for (int i = 0; i < H; i++) Y[(size_t)t * H + i] *= scales[t];
                free(scales);
            }
            if (getenv("DBG_LAYER") && L == 0) {
                double y2 = 0; for (int i = 0; i < H; i++) y2 += (double)Y[i]*Y[i];
                fprintf(stderr, "  L0 output_proj |Y|=%.4f  Y[0..2]=%.4f %.4f\n",
                        sqrt(y2), Y[0], Y[1]);
            }
            if (getenv("DBG_CMP") && L == 3) {
                double y2 = 0; for (int i = 0; i < H; i++) y2 += (double)Y[i]*Y[i];
                fprintf(stderr, "  B L3 outproj |Y|=%.4f\n", sqrt(y2));
            }
            for (int t = 0; t < n; t++)
                for (int i = 0; i < H; i++)
                    x[(size_t)t * H + i] += Y[(size_t)t * H + i];
            if (getenv("DBG_CMP") && L == 3) {
                float xmx = 0; size_t at = 0;
                for (int t = 0; t < n; t++) for (int i = 0; i < H; i++) {
                    float a = fabsf(x[(size_t)t*H+i]);
                    if (a > xmx) { xmx = a; at = (size_t)t*H+i; }
                }
                fprintf(stderr, "  B L3 outproj后 x最大=%.4f at=%zu (t=%zu i=%zu)\n",
                        xmx, at, at/H, at%H);
            }
            free(Y);
        }
        if (getenv("DBG_LAYER")) {
            double s0 = 0, s1 = 0;
            for (int i = 0; i < H; i++) {
                s0 += (double)x[i] * x[i];
                s1 += (double)x[(size_t)H + i] * x[(size_t)H + i];
            }
            fprintf(stderr, "B%02d attn后 |x0|=%.5f |x1|=%.5f\n", L, sqrt(s0), sqrt(s1));
        }
        if (getenv("DBG_STAGE")) fprintf(stderr, "[stage] L%d outproj后\n", L);
        free(attnout);

        // MLP
        snprintf(name, sizeof name, "blk.%d.ffn_norm.weight", L);
        GgufTensor *norm2 = gguf_find(gm, name);
        if (!norm2) break;
        nw = malloc(H * sizeof(float));
        tensor_to_f32(norm2, nw, H);
        rmsnorm_n(x, nw, h, n, H, (float)gm->rms_eps);
        free(nw);

        if (getenv("DBG_STAGE")) fprintf(stderr, "[stage] L%d ffn_norm后\n", L);
        snprintf(name, sizeof name, "blk.%d.ffn_gate.weight", L); GgufTensor *fg = gguf_find(gm, name);
        snprintf(name, sizeof name, "blk.%d.ffn_up.weight", L);   GgufTensor *fu = gguf_find(gm, name);
        snprintf(name, sizeof name, "blk.%d.ffn_down.weight", L); GgufTensor *fd = gguf_find(gm, name);

        quantize_n(h, qa, bsa, n, H, qa_stride, bs_stride);
        // 同样：量化只做一次，scale 留着反量化用
        float *mfs = calloc(n, sizeof(float));
        for (int t = 0; t < n; t++) {
            float mxv = 0.f;
            for (int i = 0; i < H; i++) {
                float a = fabsf(h[(size_t)t * H + i]);
                if (a > mxv) mxv = a;
            }
            mfs[t] = (mxv > 1e-8f) ? mxv / 127.0f : 1.0f;
        }
        if (getenv("DBG_CMP") && L == 3) {
            float hm = 0; for (int i = 0; i < H; i++) { float a = fabsf(h[i]); if (a > hm) hm = a; }
            fprintf(stderr, "  B L3 ffn_norm h最大=%.4f mfs[0]=%.6f mfs[1]=%.6f\n", hm, mfs[0], n>1?mfs[1]:0.f);
        }
        float *fgY = calloc((size_t)n * FF, sizeof(float));
        float *fuY = calloc((size_t)n * FF, sizeof(float));
        // 按类型分流：per-32 只有 Q4_K（type 12）支持，
        // ffn_up 有 4 层是 Q5_K(13)、3 层是 Q6_K(14)，必须走 per-tensor。
        const int fg32 = (fg && fg->type == 12);
        const int fu32 = (fu && fu->type == 12);
        if (fg32 || fu32) {
            quantize_n32(h, qa32, bsa32, asc, n, H, qa_stride, bs_stride, asc_stride);
        }
        float *mfsg = NULL, *mfsu = NULL;
        if (fg32) gemm_packed_asc(fg, qa32, bsa32, asc, qa_stride, bs_stride, asc_stride, fgY, n, (size_t)H, (size_t)FF);
        else {
            quantize_n(h, qa, bsa, n, H, qa_stride, bs_stride);
            gemm_packed(fg, qa, bsa, qa_stride, bs_stride, fgY, n, (size_t)H, (size_t)FF);
            mfsg = calloc(n, sizeof(float));
            for (int t = 0; t < n; t++) {
                float mxv = 0.f;
                for (int i = 0; i < H; i++) { float a = fabsf(h[(size_t)t*H+i]); if (a > mxv) mxv = a; }
                mfsg[t] = (mxv > 1e-8f) ? mxv / 127.0f : 1.0f;
                for (int i = 0; i < FF; i++) fgY[(size_t)t*FF+i] *= mfsg[t];
            }
        }
        if (fu32) gemm_packed_asc(fu, qa32, bsa32, asc, qa_stride, bs_stride, asc_stride, fuY, n, (size_t)H, (size_t)FF);
        else {
            if (!fg32) quantize_n(h, qa, bsa, n, H, qa_stride, bs_stride);
            gemm_packed(fu, qa, bsa, qa_stride, bs_stride, fuY, n, (size_t)H, (size_t)FF);
            mfsu = calloc(n, sizeof(float));
            for (int t = 0; t < n; t++) {
                float mxv = 0.f;
                for (int i = 0; i < H; i++) { float a = fabsf(h[(size_t)t*H+i]); if (a > mxv) mxv = a; }
                mfsu[t] = (mxv > 1e-8f) ? mxv / 127.0f : 1.0f;
                for (int i = 0; i < FF; i++) fuY[(size_t)t*FF+i] *= mfsu[t];
            }
        }
        free(mfsg); free(mfsu);
        if (getenv("DBG_CMP") && L == 3) {
            fprintf(stderr, "  B L3 fgY[0..3]=%.1f %.1f  fuY[0..3]=%.1f %.1f\n",
                    fgY[0], fgY[1], fuY[0], fuY[1]);
        }
        if (getenv("DBG_STAGE")) fprintf(stderr, "[stage] L%d fgfu后\n", L);
        for (int t = 0; t < n; t++) {
            const float sf = 1.0f;   // scale 已在上面按类型分别乘过
            for (int i = 0; i < FF; i++) {
                const float g = fgY[(size_t)t * FF + i] * sf;
                const float u = fuY[(size_t)t * FF + i] * sf;
                const float th = tanhf(0.7978845608f * (g + 0.044715f * g * g * g));
                fgY[(size_t)t * FF + i] = 0.5f * g * (1.0f + th) * u;
            }
        }
        free(fuY); free(mfs);

        if (getenv("DBG_STAGE")) fprintf(stderr, "[stage] L%d down前\n", L);
        {
            float *dsc = calloc(n, sizeof(float));
            for (int t = 0; t < n; t++)
                dsc[t] = quantize_act(fgY + (size_t)t * FF,
                                      qa + (size_t)t * qa_stride,
                                      bsa + (size_t)t * bs_stride, FF);
            float *Y = calloc((size_t)n * H, sizeof(float));
            gemm_packed(fd, qa, bsa, qa_stride, bs_stride, Y, n, (size_t)FF, (size_t)H);
            for (int t = 0; t < n; t++)
                for (int i = 0; i < H; i++)
                    x[(size_t)t * H + i] += Y[(size_t)t * H + i] * dsc[t];
            free(Y); free(dsc);
        }
        if (getenv("DBG_CMP") && L == 3) {
            float fm = 0; for (int t = 0; t < n; t++) for (int i = 0; i < FF; i++) {
                float a = fabsf(fgY[(size_t)t*FF+i]);
                if (a > fm) fm = a;
            }
            fprintf(stderr, "  B L3 fgY最大=%.4e\n", fm);
        }
        free(fgY);
        if (getenv("DBG_CMP")) {
            const size_t lo = (size_t)(n - 1) * H;
            double n2 = 0; for (int i = 0; i < H; i++) n2 += (double)x[lo + i]*x[lo + i];
            fprintf(stderr, "B%02d last|x|=%.5f x0=%.5f\n", L, sqrt(n2), x[lo]);
        }
    }

    // 最终 norm + lm_head：所有位置都要 logits。投机验证必须逐位确认 draft。
    //
    // 【坑】不能直接 gemv_rows(..., out + t*e->V, 0, V) —— 传偏移指针时
    // gemv_rows 内部的多线程分段写不生效（实测输出全是 0）。
    // 改成每次写进独立的 tmp，再 memcpy 到 out 的对应位置。
    float *out = calloc((size_t)n * e->V, sizeof(float));
    if (out) {
        float *hx = calloc(H, sizeof(float));
        float *tmp = malloc((size_t)e->V * sizeof(float));
        int8_t *qx = malloc(H);
        int16_t *qxbs = malloc((H / 16 + 1) * 2);
        GgufTensor *on = gguf_find(gm, "output_norm.weight");
        for (int t = 0; t < n; t++) {
            if (on) {
                float *nw2 = malloc(H * sizeof(float));
                tensor_to_f32(on, nw2, H);
                rmsnorm(x + (size_t)t * H, nw2, hx, H, (float)gm->rms_eps);
                free(nw2);
            } else {
                memcpy(hx, x + (size_t)t * H, H * sizeof(float));
            }
            const float sl = quantize_act(hx, qx, qxbs, H);
            gemv_rows(emb, qx, qxbs, tmp, 0, (size_t)e->V);
            for (int i = 0; i < e->V; i++) out[(size_t)t * e->V + i] = tmp[i] * sl;
        }
        free(qx); free(qxbs); free(hx); free(tmp);
    }

    e->pos = pos0 + n;
    free(x); free(qa); free(bsa); free(qkv); free(h);
    return out;
}
