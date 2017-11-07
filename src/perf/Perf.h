#ifndef __PERF_H__
#define __PERF_H__

int CudaQR(int m, int n, double *runtime_ms);
int TileQR(int m, int n, double *runtime_ms);

int RunAll();

#endif