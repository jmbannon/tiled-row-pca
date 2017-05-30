#include "Vector.h"
#include "Matrix.h"
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

/**
  * Wrapper for single-threaded house qr kernel. Use only for testing.
  */
#ifdef __cplusplus
extern "C"
#endif
int
Block_house_qr(cublasHandle_t *handle, Matrix *A);

/**
  * Wrapper for single-threaded dgeqt2 kernel. Use only for testing.
  */
#ifdef __cplusplus
extern "C"
#endif
int
Block_dgeqt2(cublasHandle_t *handle, Matrix *A, Matrix *T);



#ifdef __cplusplus
extern "C"
#endif
int
TileQR_cublasDgemm_hmn(cublasDiagType_t diag,
                       int m, int n, int k,
                       const Numeric alpha,
                       const Numeric *A, int lda,
                       const Numeric *B, int ldb,
                       Numeric *C, int ldc);


#ifdef __cplusplus
extern "C"
#endif
int
TileQR_cublasDgemm_mht(cublasDiagType_t diag,
                       int m, int n, int k,
                       const Numeric alpha,
                       const Numeric *A, int lda,
                       const Numeric *B, int ldb,
                       Numeric *C, int ldc);

#endif
