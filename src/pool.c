// 持久线程池 —— 用原子变量 + 自旋实现屏障。
//
// 【为什么不用 condvar】GEMV 每层调 5 次、36 层共 180 次，
// 每次 pthread_cond_broadcast + wait 都是系统调用（~10µs），
// 180 次就是 1.8ms，实测更差（约 25ms 归因不明，很可能就在这）。
// 用原子计数器自旋：worker 空转等待的时间远小于线程被唤醒的延迟，
// 而且 3 个线程跑满 4 个大核时本来就有空闲核。
#include "pool.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>

#define MAXT 8

static pthread_t        g_th[MAXT];
static int              g_n = 0;
static int              g_started = 0;

static volatile unsigned long g_seq = 0;      // 任务代际
static volatile unsigned long g_seen[MAXT];   // 各 worker 已处理的代际
static volatile int           g_done = 0;     // 本轮完成数
static volatile int           g_quit = 0;
static PoolJob                g_job;

static inline void cpu_relax(void) {
#if defined(__aarch64__)
    __asm__ __volatile__("yield");
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}

static void *worker(void *arg) {
    const int id = (int)(long)arg + 1;
    unsigned long last = 0;
    for (;;) {
        if (g_quit) return NULL;
        if (g_seen[id] == last) {
            // 还没新任务：自旋一会儿再让出
            // 自旋 50 次后让出。纯自旋（不 yield）实测会段错误：
            // worker 一直占着 CPU，和主线程抢 g_done 的写回时序。
            for (int s = 0; s < 50 && g_seen[id] == last && !g_quit; s++) cpu_relax();
            if (g_seen[id] == last) { sched_yield(); continue; }
        }
        last = g_seen[id];
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        PoolJob job = g_job;
        job.fn(job.arg, id);
        __atomic_add_fetch(&g_done, 1, __ATOMIC_RELEASE);
    }
}

void pool_init(int nthreads) {
    if (nthreads < 1) nthreads = 1;
    if (nthreads > MAXT) nthreads = MAXT;
    if (g_started) return;
    g_n = nthreads;
    g_quit = 0; g_done = 0; g_seq = 0;
    memset((void *)g_seen, 0, sizeof g_seen);
    if (nthreads == 1) { g_started = 1; return; }
    for (int i = 0; i < nthreads - 1; i++)
        pthread_create(&g_th[i], NULL, worker, (void *)(long)i);
    g_started = 1;
}

void pool_shutdown(void) {
    if (!g_started) return;
    if (g_n > 1) {
        g_quit = 1;
        for (int i = 0; i < g_n - 1; i++) pthread_join(g_th[i], NULL);
    }
    g_started = 0;
}

int pool_threads(void) { return g_n; }

void pool_run(PoolJob *job, int nworkers) {
    (void)nworkers;
    if (!g_started) return;
    if (g_n <= 1) { job->fn(job->arg, 0); return; }

    g_job = *job;
    __atomic_thread_fence(__ATOMIC_RELEASE);
    g_done = 0;
    unsigned long seq = ++g_seq;
    __atomic_store_n(&g_seen[1], seq, __ATOMIC_RELEASE);
    if (g_n > 2) __atomic_store_n(&g_seen[2], seq, __ATOMIC_RELEASE);
    if (g_n > 3) __atomic_store_n(&g_seen[3], seq, __ATOMIC_RELEASE);

    job->fn(job->arg, 0);

    const int need = g_n - 1;
    while (__atomic_load_n(&g_done, __ATOMIC_ACQUIRE) < need) {
        for (int s = 0; s < 100; s++) cpu_relax();
    }
}
