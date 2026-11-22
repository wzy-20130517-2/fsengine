// 逐步对比：每步打印 argmax 和 top-2 的差距，看精度崩溃的临界点
#include "engine.h"
#include <stdio.h>
#include <stdlib.h>
static void top2(const float *v, int n, int *i1, int *i2, float *d) {
    int a=0,b=-1; float va=v[0], vb=-1e30f;
    for (int i=1;i<n;i++){ if(v[i]>va){b=a;vb=va;a=i;va=v[i];} else if(v[i]>vb){b=i;vb=v[i];} }
    *i1=a;*i2=b;*d=va-vb;
}
int main(int c, char **v) {
    Engine *e = engine_open_t(v[1], 256, 4);
    int prompt[5] = {1046, 11862, 314, 17349, 390};
    int want[8] = {19500, 27, 19500, 390, 259, 5508, 6594, 304};
    float *lg=NULL;
    for (int i=0;i<5;i++) lg = engine_step(e, prompt[i]);

    int feed=19500;
    for (int i=0;i<8;i++){
        int i1,i2; float d;
        top2(lg, e->V, &i1, &i2, &d);
        printf("步%d: 我=%d  期望=%d  %s   top2差=%.4f (第二=%d)\n",
               i, i1, want[i], i1==want[i]?"✓":"✗", d, i2);
        if (i<7) { feed = i1; lg = engine_step(e, feed); }
    }
    return 0;
}
