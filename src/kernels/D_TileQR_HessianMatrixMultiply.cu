#include "../TileQR_Operations.h"
#include "../constants.h"
#include "../error.h"
#include "../Vector.h"
#include "../Matrix.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdbool.h>

/**
 * Multiplies a m-by-k Hessianberg matrix with a k-by-n matrix.
 * Multiplies its triangle portion using optimized triangle-matrix multiply.
 *
 * C = alpha * H * B
 *
 * @param handle cuBLAS handle
 * @param diag Whether the diagonal in the Hessian is Unit (1's) or Non-Unit.
 * @param m Rows of Hessianberg matrix H and C
 * @param n Cols of B and C
 * @param k Cols of Hessianberg matrix H and rows of B
 * @param alpha Scalar
 * @param A m-by-k Hessianberg matrix
 * @param B k-by-n Matrix
 * @param C m-by-n Matrix
 */
__device__ int cublasDgemm_hmn(cublasHandle_t handle,
                               cublasDiagType_t diag,
                               int m, int n, int k,
                               const Numeric alpha,
                               const Numeric *A, int lda,
                               const Numeric *B, int ldb,
                               Numeric *C, int ldc)
{

  int res;
  Numeric zero = 0.0;
  // Multiply the triangular portion
  #if FLOAT_NUMERIC
    res = cublasStrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, diag, k, n, &alpha, A, lda, B, ldb, C, ldc);
  #else
    res = cublasDtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, diag, k, n, &alpha, A, lda, B, ldb, C, ldc);
  #endif 
  CHECK_CUBLAS_RETURN(res, "Triangle portion of Hessianberg matrix multiply failed");

  if (m != k) {
  	// Multiply the rectangular portion if the Hessianberg matrix H is not a triangular matrix
    #if FLOAT_NUMERIC
      res = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m - k, n, k, &alpha, &A[k], lda, B, ldb, &zero, &C[k], ldc);
    #else
      res = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m - k, n, k, &alpha, &A[k], lda, B, ldb, &zero, &C[k], ldc);
    #endif
    CHECK_CUBLAS_RETURN(res, "Rectangle portion of Hessianberg matrix multiply failed");
  }
  return 0;
}

/**
 * Multiplies a m-by-k matrix with a n-by-k transposed Hessianberg matrix.
 * Multiplies its triangle portion using optimized triangle-matrix multiply.
 *
 * C = alpha * A * t(H)
 *
 * @param handle cuBLAS handle
 * @param diag Whether the diagonal in the Hessian is Unit (1's) or Non-Unit.
 * @param m Rows of matrix A and C
 * @param n Cols of t(H) and C (rows of H)
 * @param k Cols of A and rows of t(H) (or cols of H)
 * @param alpha Scalar
 * @param A m-by-k Matrix
 * @param H n-by-k Hessianberg matrix to be multiplied transposed
 * @param C m-by-n Matrix
 */
__device__ int cublasDgemm_mht(cublasHandle_t handle,
                               cublasDiagType_t diag,
                               int m, int n, int k,
                               const Numeric alpha,
                               const Numeric *A, int lda,
                               const Numeric *B, int ldb,
                               Numeric *C, int ldc)
{
  int res;
  Numeric zero = 0.0;
  // Multiply the triangular portion
  #if FLOAT_NUMERIC
    res = cublasStrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, diag, m, k, &alpha, B, ldb, A, lda, C, ldc);
  #else
    res = cublasDtrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, diag, m, k, &alpha, B, ldb, A, lda, C, ldc);
  #endif 
  CHECK_CUBLAS_RETURN(res, "Triangle portion of Hessianberg matrix multiply failed");

  if (n != k) {
  	// Multiply the rectangular portion if the Hessianberg matrix H is not a triangular matrix
    int C_rectangle_idx = MAT_POS(0, k, ldc);
    #if FLOAT_NUMERIC
      res = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, lda, &B[k], ldb, &zero, &C[C_rectangle_idx], ldc);
    #else
      res = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, lda, &B[k], ldb, &zero, &C[C_rectangle_idx], ldc);
    #endif
    CHECK_CUBLAS_RETURN(res, "Rectangle portion of Hessianberg matrix multiply failed");
  }

  return 0;
}

// Wrapper kernel to cublasDgemm_hmn. Use only for testing.
__global__ void TileQR_cublasDgemm_hmn_kernel(cublasDiagType_t diag,
                                              int m, int n, int k,
                                              const Numeric alpha,
                                              const Numeric *A, int lda,
                                              const Numeric *B, int ldb,
                                              Numeric *C, int ldc)
{
    cublasHandle_t handle;
    int res = cublasCreate(&handle);

    cublasDgemm_hmn(handle, diag, m, n, k, alpha, A, lda, B, ldb, C, ldc);
}

// Wrapper function to single-threaded cublasDgemm_hmn kernel. Use only for testing.
extern "C"
int
TileQR_cublasDgemm_hmn(cublasDiagType_t diag,
                       int m, int n, int k,
                       const Numeric alpha,
                       const Numeric *A, int lda,
                       const Numeric *B, int ldb,
                       Numeric *C, int ldc)
{    
    TileQR_cublasDgemm_hmn_kernel<<<1, 1>>>(diag, m, n, k, alpha, A, lda, B, ldb, C, ldc);
    return 0;
}

// Wrapper kernel to cublasDgemm_mht. Use only for testing.
__global__ void TileQR_cublasDgemm_mht_kernel(cublasDiagType_t diag,
                                              int m, int n, int k,
                                              const Numeric alpha,
                                              const Numeric *A, int lda,
                                              const Numeric *B, int ldb,
                                              Numeric *C, int ldc)
{
    cublasHandle_t handle;
    int res = cublasCreate(&handle);

    cublasDgemm_mht(handle, diag, m, n, k, alpha, A, lda, B, ldb, C, ldc);
}

// Wrapper function to single-threaded cublasDgemm_mht kernel. Use only for testing.
extern "C"
int
TileQR_cublasDgemm_mht(cublasDiagType_t diag,
                       int m, int n, int k,
                       const Numeric alpha,
                       const Numeric *A, int lda,
                       const Numeric *B, int ldb,
                       Numeric *C, int ldc)
{    
    TileQR_cublasDgemm_mht_kernel<<<1, 1>>>(diag, m, n, k, alpha, A, lda, B, ldb, C, ldc);
    return 0;
}