#ifndef __PERF_H__
#define __PERF_H__

int TileQR_20k_4k(double *runtime_ms);
int CudaQR_20k_4k(double *runtime_ms);

int RunAll();

#endif