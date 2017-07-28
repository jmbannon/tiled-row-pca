#include "Vector.h"
#include "Matrix.h"
#include "BlockMatrix.h"
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
TileQR_house(cublasHandle_t *handle, Vector *in, Vector *out);

/**
  * Wrapper for single-threaded dgeqt2 kernel. Use only for testing.
  */
#ifdef __cplusplus
extern "C"
#endif
int
TileQR_blk_dgeqt2(Matrix *A, Matrix *T);

#ifdef __cplusplus
extern "C"
#endif
int
TileQR_house_qr_q(Matrix *Y,
                  Matrix *T,
                  Matrix *Q,
                  Matrix *Q_,
                  int m, int n);


#ifdef __cplusplus
extern "C"
#endif
int
BlockMatrix_TileQR_single_thread(BlockMatrix *A);

#ifdef __cplusplus
extern "C"
#endif
int
BlockMatrix_TileQR_multi_thread(BlockMatrix *A);

#endif
