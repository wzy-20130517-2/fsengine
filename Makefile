# fsengine — 手机端 LLM 推理引擎
#
# 编译参数都是实测选出来的：
#   -march=armv8.2-a+dotprod+fp16  A77 支持 SDOT 和 FP16，不支持 i8mm/SVE
#   -O3                            -Ofast 只快 0.2% 但改变浮点语义，不划算
#   -flto                          实测无收益，无害，保留
CC      := gcc
ARCH    := -march=armv8.2-a+dotprod+fp16
CFLAGS  := -O3 $(ARCH) -Isrc -flto
LDFLAGS := -pthread -flto -lm

SRCS := src/gguf.c src/dequant.c src/gemv.c src/tensor.c src/pool.c \
        src/forward.c src/repack.c src/x8cache.c src/i8cache.c src/quant.c \
        src/tok.c src/tok_encode.c
OBJS := $(SRCS:.c=.o)

all: server

# 头文件依赖：改了 .h 必须重编对应的 .o，否则结构体布局变了会段错误（踩过）
HDRS := src/gguf.h src/tensor.h src/pool.h src/engine.h src/repack.h \
        src/x8cache.h src/i8cache.h src/quant.h src/tok.h

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $@

# HTTP 服务：兼容 llama-server 的 /health、/completion、/v1/chat/completions
server: $(OBJS) fsengine_server.c
	$(CC) $(CFLAGS) -o fsengine-server fsengine_server.c $(OBJS) $(LDFLAGS)

# 测速：每轮重置 pos 并清 KV cache，5 轮取中位
bench: $(OBJS) bench_stable2.c
	$(CC) $(CFLAGS) -o bench_stable2 bench_stable2.c $(OBJS) $(LDFLAGS)

# 精度验证：和 llama.cpp 逐 token 对比 argmax
verify: $(OBJS) cmp_seq.c
	$(CC) $(CFLAGS) -o cmp_seq cmp_seq.c $(OBJS) $(LDFLAGS)

clean:
	rm -f src/*.o fsengine-server bench_stable2 cmp_seq

.PHONY: all server bench verify clean
