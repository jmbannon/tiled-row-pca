#include "Vector.h"
#include <cublas_v2.h>

#ifndef BLOCK_QR_OPERATIONS_H_
#define BLOCK_QR_OPERATIONS_H_

/**
  * Wrapper for single-threaded house kernel. Use only for testing.
  */
#ifdef __cplusplus
extern "C"
#endif
int
Block_house(cublasHandle_t *handle, Vector *in, Vector *out);

#endif
