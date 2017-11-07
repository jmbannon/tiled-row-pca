#include "Perf.h"
#include <stdio.h>

#define MAX_NUM_PERF_TESTS 100


typedef struct _PerfTest {
    int m;
    int n;
} PerfTest;

PerfTest perfTests[MAX_NUM_PERF_TESTS];
int perfCount = 0;

void addPerfTest(int m, int n)
{
    PerfTest test = { .m = m, .n = n};
    perfTests[perfCount++] = test;
}

void addAllPerfTests() {
    addPerfTest(1000, 1000);
    addPerfTest(1000, 2000);
    addPerfTest(1000, 4000);
    addPerfTest(1000, 8000);
    addPerfTest(1000, 16000);
    //addPerfTest(1000, 32000);

    addPerfTest(1000, 1000);
    addPerfTest(2000, 1000);
    addPerfTest(4000, 1000);
    addPerfTest(8000, 1000);
    addPerfTest(16000, 1000);
    //addPerfTest(32000, 1000);

    addPerfTest(4000, 1000);
    addPerfTest(4000, 2000);
    addPerfTest(4000, 4000);
    addPerfTest(4000, 8000);
    //addPerfTest(4000, 16000);

    addPerfTest(1000, 4000);
    addPerfTest(2000, 4000);
    addPerfTest(4000, 4000);
    addPerfTest(8000, 4000);
    //addPerfTest(16000, 4000);
}

int RunAll() {
    addAllPerfTests();

    int res = 0;
    double runtime_tile = 0;
    double runtime_cuda = 0;
    int m, n;
    for (int i = 0; i < perfCount; i++) {
        m = perfTests[i].m;
        n = perfTests[i].n;
        res = TileQR(m, n, &runtime_tile);
        if (res != 0) {
            runtime_tile = -1;
        }

        res = CudaQR(m, n, &runtime_cuda);
        if (res != 0) {
            runtime_cuda = -1;
        }

        printf("%7.3f, %7.3f, %d, %d\n", runtime_tile, runtime_cuda, m, n);
        
    }
    printf("\n");
    return 0;
}