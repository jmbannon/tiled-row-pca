#include "BlockMatrix.h"
#include "Vector.h"

#ifndef __TMP_CUDA_H__
#define __TMP_CUDA_H__

#ifdef __cplusplus
extern "C"
#endif
void someFunction(void);

#ifdef __cplusplus
extern "C"
#endif
int matrixColumnSums(BlockMatrix *in, Vector *out, double scalar);

#endif
