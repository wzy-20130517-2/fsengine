// 持久线程池：GEMV 一个 token 要调 180 次，每次创建/销毁 4 个线程
// 约 30µs，加起来 22ms —— 占每轮 250ms 的 9%。
// 改成常驻线程 + 屏障同步，这部分开销基本归零。
#ifndef FSENGINE_POOL_H
#define FSENGINE_POOL_H

#include <stddef.h>

// 任务函数：在 worker 线程里执行
typedef void (*pool_fn)(void *arg, int worker_id);

void pool_init(int nthreads);
void pool_shutdown(void);
int  pool_threads(void);

// 把 n 份任务并行跑完再返回。fn 会被每个 worker 调用一次，
// 参数 arg 由调用方共享，worker_id 是 0..nthreads-1。
// caller 也参与干活（id 0 由调用线程执行），省一个线程。
typedef struct {
    pool_fn fn;
    void   *arg;
} PoolJob;

void pool_run(PoolJob *job, int nworkers);

#endif
