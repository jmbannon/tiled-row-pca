#include "Perf.h"
#include <stdio.h>

#define MAX_NUM_PERF_TESTS 100


typedef struct _PerfTest {
    char *name;
    int (*testFunction)(double *);
} PerfTest;

PerfTest perfTests[MAX_NUM_PERF_TESTS];
int perfCount = 0;

void addPerfTest(char *name, 
             int (*testFunction)(double *))
{
    PerfTest test = { .name = name, .testFunction = testFunction };
    perfTests[perfCount++] = test;
}

void addAllPerfTests() {
    addPerfTest("TileQR_1000_1000", TileQR_1000_1000);
}

int RunAll() {
    addAllPerfTests();

    int res = 0;
    double runtime_sec = 0;
    for (int i = 0; i < perfCount; i++) {
        res = (*perfTests[i].testFunction)(&runtime_sec);
        if (res != 0) {
            printf("FAILURE  -  %s\n", perfTests[i].name);
        } else {
            printf("%7.3f s -  %s\n", runtime_sec, perfTests[i].name);
        }
    }
    printf("\n");
    return 0;
}