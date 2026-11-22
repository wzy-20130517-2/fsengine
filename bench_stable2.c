// 可靠测量：热身后连测 15 轮，输出中位数和四分位
#include "engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
static int cmpd(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return x<y?-1:x>y;}
int main(int c, char **v) {
    int T = c>2?atoi(v[2]):4;
    Engine *e = engine_open_t(v[1], 512, T);
    const int N=6;
    // 【每轮必须从 pos=0 开始且 KV cache 清零】
    // decode 的 attention 成本随位置线性增长，不清零会导致每轮计算量不同，
    // 测出来就是大幅波动（实测 3.8~4.6 tok/s 乱跳）。
    const size_t kvrow = (size_t)e->NKV * e->HD;
    for(int k=0;k<N;k++) engine_step(e, k);   // 热身
    const int R=5; double ts[R];
    for(int r=0;r<R;r++){
        e->pos = 0;
        for (int L = 0; L < e->L; L++) {
            memset(e->kcache[L], 0, (size_t)e->max_ctx * kvrow * sizeof(__fp16));
            memset(e->vcache[L], 0, (size_t)e->max_ctx * kvrow * sizeof(__fp16));
        }
        double t0=now();
        for(int k=0;k<N;k++) engine_step(e, k);
        ts[r]=(now()-t0)/N;
        usleep(2000000);   // 2 秒冷却：实测连续跑 12 轮后 CPU 降频，速度从 4.5 掉到 1.0
    }
    qsort(ts,R,sizeof(double),cmpd);
    printf("T=%d  中位 %.3f tok/s  (四分位 %.3f ~ %.3f)  最优 %.3f\n",
           T, 1/ts[R/2], 1/ts[R*3/4], 1/ts[R/4], 1/ts[0]);
    return 0;
}
