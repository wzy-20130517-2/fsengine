# fsengine

手机端 LLM 推理引擎。纯 C，只依赖 NEON，为 Spark 2.5 架构手写。

不是 llama.cpp 的封装，也不基于 ggml —— 从 GGUF 解析到 kernel 全部重写。

**一句话结论：在 Redmi Note 15 Pro 上跑到内存带宽的 94%，再往上只能换硬件或压模型。**

## 实测数据

| 项目 | 数值 |
|---|---|
| 模型 | Spark-X2.6-4B-Q4_K_M（2.62 GB） |
| 生成速度 | **4.85 tok/s**（206 ms/token） |
| 有效带宽 | **12.7 GB/s** |
| 内存带宽实测上限 | 13.5 GB/s（8 核并行也一样） |
| 精度 | 与 llama.cpp **逐 token argmax 完全一致（8/8）** |

测速条件：天玑 7400-Ultra，绑 cpu4-7（4×A77 @2.6GHz），3 线程，temperature 0。

## 这台机器的带宽天花板

用 8 路独立累加器 + 多线程测出来的（`bw_retest.c` 的思路，见下）：

| 块大小 | 带宽 | 层级 |
|---|---|---|
| 32 KB | 106.7 GB/s | L1 |
| 128 KB | 53.3 GB/s | L2 |
| 2 MB | 26.0 GB/s | L3 |
| 8 MB | 13.8 GB/s | DRAM |
| 512 MB | 13.5 GB/s | DRAM |

**DRAM 就是 13.5 GB/s，多线程不叠加。** 规格表写 LPDDR4X 25.6 GB/s，那是理论值，实际 CPU 侧拿不到。

单 token decode 时权重每字节只用一次，L3（2-4 MB）装不下每层的 30 MB，所以缓存帮不上忙。这是物理限制。

## 实现要点

### 量化格式

支持 GGUF 的 Q2_K / Q3_K / Q4_K / Q5_K / Q6_K 全部解量化，逐值对齐官方实现。

关键布局（踩了很久才搞对）：

- **Q4_K**：每轮 64 权重，32 字节低半字节出前 32 个，高半字节出后 32 个，`qs += 32; is += 2`
- **Q6_K**：外层 128 一组，组内 `l` 产出 `l+0/l+32/l+64/l+96`，scale 取 `sc[is+0/2/4/6]`
- **Q5_K**：`qh` 只有 32 字节覆盖全部 256 权重，每轮用不同掩码，指针不动

### 8 路交错（x8）

照 llama.cpp 的 Q4_Kx8 实现，一个 super-block 1152 字节：`d[8] + dmin[8] + scales[96] + qs[1024]`。

单张量测比逐行快约 3 倍（3.14 → 9.70 GB/s）。

**Q6_K 的 x8 也实现了，但实测比逐行慢**（3.0 vs 4.2 GB/s），原因是每个 block 要展开 128 个 int16 scale，抵消了 vdot 的收益。默认不启用。

### 双套激活量化

这是精度的关键，单 token 路径同时存在两套：

- **per-32**（`quant_act_block`）：给普通 GEMV 用，误差 0.0019
- **per-256**（`quant_act_q8k`）：给 x8 内核用（x8 要求整个 256-block 共用一份 scale）

用 per-tensor（整段一个 scale）会让第 3 个 token 开始 argmax 翻转（误差 0.00425 正好等于 top1/top2 差距）。

### 自旋线程池

`pthread_barrier_t` 在 Android bionic 上死锁，condvar 每次 `pool_run` 要 0.68 μs。改成自旋原子变量（`g_seq/g_seen/g_done` + `yield`）后省掉每 token 216 次系统调用。

纯自旋（不 yield）会段错误 —— worker 一直占 CPU 与主线程抢 `g_done` 时序。

## HTTP 服务

`fsengine-server` 实现 llama-server 的三个接口，可以直接给 [ccmchat-cli](https://github.com/wzy-20130517-2/Claude-Code-Mobile) 用：

- `GET /health`
- `POST /completion`（SSE）
- `POST /v1/chat/completions`（SSE，支持 `chat_template_kwargs.enable_thinking`）

```bash
make server
./fsengine-server -m 模型.gguf -c 512 -t 3 --port 8931
```

参数和 llama-server 同名（`-m/-c/-t/--port`），可以互换。

### 分词器

从 GGUF 里读 `tokenizer.ggml.tokens` / `merges`，自己实现 GPT-2 风格 BPE。

**两个坑**：

1. **特殊 token 必须整体匹配**。`<｜end▁of▁sentence｜>` 如果让 BPE 处理，会被拆成 `<`、`end`、`of`… 8 个 token。更糟的是前面的 `.` 会和 `<` 合成一个 token，导致后面的标记完全认不出来。所以要在 BPE 之前先按特殊 token 切开。

2. **`▁`（U+2581）要映射成空格**。这是 SentencePiece/GPT-2 词表表示空格的约定，不映射的话所有带空格的 token 都找不到。

## 实测否定的方案

这些都试过，都不行，写下来免得重复踩：

| 方案 | 结果 |
|---|---|
| 投机解码（1.7B draft） | 接受率仅 15.6%，k=4 时 2.33 tok/s，比直接跑慢 |
| n-gram 投机解码 | 接受率 52-82%，但每轮只 draft 0.3-0.5 个，验证成本吃掉收益，0.81-0.95x |
| Q6_K 的 x8 | 单张量 1.16x，接进引擎无收益 |
| int8 预解包缓存 | 多读 22% 字节，整体更慢 |
| mlock 常驻缓存 | 无效 —— 模型本来就在 page cache 里，没被换出 |
| `-Ofast` | 只快 0.2%，改变浮点语义 |
| 手写汇编 | 检查过寄存器溢出，只有 prologue 的 2 处，没有优化空间 |
| 绑定小核 / 全 8 核 | 比只绑大核慢 |

## 为什么不做这些

- **MMA / i8mm**：Cortex-A77 没有这个扩展，实测执行 `SMMLA` 直接 SIGILL
- **T-MAC 的 LUT 查表**：优势在 1-2 bit（LUT 只有 4 项）。4-bit 时每字节要查 2 次表，而 1 条 vdot 就能算完 16 字节
- **无损压缩（Cloudflare Unweight / DFloat11）**：它们压的是 BF16 权重，Q4_K 是 4.5 bit/权重，元数据只占 11%，压不出 30%
- **BitNet 三元模型**：需要原生三元训练的模型，Q4_K 转不了

## 精度验证

`cmp_seq` 和 llama.cpp 逐 token 对比同一个 prompt：

```
prompt: [1046, 11862, 314, 17349, 390]
期望:   [19500, 27, 19500, 390, 259, 5508, 6594, 304]
```

8 个 token 的 argmax 全部命中。

## 文件结构

```
fsengine_server.c     HTTP 服务（兼容 llama-server 接口）
bench_stable2.c       测速
cmp_seq.c             精度验证
src/
  gguf.c              GGUF 解析（mmap 零拷贝）
  dequant.c           各量化格式解量化
  gemv.c              逐行融合内核
  repack.c            x8 交错重排 + 8 路内核
  tensor.c            张量到 GEMV 的分派层
  forward.c           完整前向（36 层 SWA/Full、GeGLU、RoPE、KV cache）
  pool.c              自旋线程池
  quant.c             激活量化（per-32 / per-256）
  x8cache.c           x8 重排缓存管理
  tok.c / tok_encode.c BPE 分词器
```

## 已知问题

- 分词器的预切分规则（`tokenizer.ggml.pre = spark2_5`）是猜的，英文会把 `helpful` 拆成 `help`+`ful`。中文场景影响小。
- 不支持批量 prefill，prompt 是一个 token 一个 token 跑的。
- 模型加载时 x8 重排要额外 1.4 GB 内存（热张量复制一份），机器内存紧张时会挤爆 page cache。

## 编译

```bash
make server    # fsengine-server
make bench     # bench_stable2
make verify    # cmp_seq
```

依赖：gcc、NEON（armv8.2-a + dotprod + fp16）。没有第三方库。

## 许可

MIT
