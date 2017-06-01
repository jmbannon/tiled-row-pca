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
  cudaDeviceSynchronize();

  if (n != k) {
  	// Multiply the rectangular portion if the Hessianberg matrix H is not a triangular matrix
    int C_rectangle_idx = MAT_POS(0, k, ldc);
    #if FLOAT_NUMERIC
      res = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, lda, &B[k], ldb, &zero, &C[C_rectangle_idx], ldc);
    #else
      res = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, A, lda, &B[k], ldb, &zero, &C[C_rectangle_idx], ldc);
    #endif
    CHECK_CUBLAS_RETURN(res, "Rectangle portion of Hessianberg matrix multiply failed");
    cudaDeviceSynchronize();
  }

  return 0;
}


/**
  * Computes Q = I + (Y * T * t(Y))
  *
  * @param handle cuBLAS handle
  * @param Y m-by-n Hessianberg matrix containing householder vectors in lower portion.
  * @param T n-by-n Upper-triangular matrix
  * @param Q_ m-by-n work matrix to store Y * T
  * @return Q m-by-m output matrix
  */
__device__ int house_qr_q(cublasHandle_t *handle, Numeric *Y, Numeric *T, Numeric *Q, Numeric *Q_, int m, int n)
{
    int res;
    Numeric alpha = 1.0;
    
    // Calculates Q' = Y * T
    res = cublasDgemm_hmn(*handle, CUBLAS_DIAG_UNIT, m, n, n, alpha, Y, m, T, n, Q_, m);
    CHECK_CUBLAS_RETURN(res, "Failed to compute Q' = Y * T");
    cudaDeviceSynchronize();

    // Calculates Q = Q' * t(Y)
    //              = Y * T * t(Y)
    res = cublasDgemm_mht(*handle, CUBLAS_DIAG_UNIT, m, m, n, alpha, Q_, m, Y, m, Q, m);
    CHECK_CUBLAS_RETURN(res, "Failed to compute Q = Q' * t(Y)");
    cudaDeviceSynchronize();

    // Calculates Q = I + Q
    //              = I + (Y * T * t(Y))
    for (int i = 0; i < m; i++) {
      Q[MAT_POS(i, i, m)] += 1.0;
    }

    return 0;
}


/**
  * Multiplies a matrix A s.t. A = t(Q) * A
  *                              = t(I + (Y * T * t(Y))) * A
  * where Q is from a diagonal tile P, where P = QR, and A is an adjacent tile to the right of P.
  *
  * @param handle cuBLAS handle
  * @param A m-by-n matrix to multiply and override. Adjacent to the source tile of Q.
  * @param Y m-by-n Hessianberg matrix holding householder vectors.
  * @param T n-by-n matrix.
  * @param Q m-by-m matrix to store Q.
  * @param Q_ m-by-n work matrix.
  */
__device__ int TileQR_dlarfb(cublasHandle_t *handle, Numeric *A, Numeric *Y, Numeric *T, Numeric *Q, Numeric *Q_, int m, int n)
{
  int res;

  res = house_qr_q(handle, Y, T, Q, Q_, m, n);
  CHECK_ZERO_ERROR_RETURN(res, "Failed to compute house_qr_q");

  Numeric zero = 0.0;
  Numeric alpha = 1.0;
  #if FLOAT_NUMERIC
    res = cublasSgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, m, &alpha, Q, m, A, m, &zero, Q_, m);
  #else
    res = cublasDgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, m, &alpha, Q, m, A, m, &zero, Q_, m);
  #endif
  CHECK_CUBLAS_RETURN(res, "Failed to compute Q' = t(Q) * A")


  #if FLOAT_NUMERIC
    res = cublasScopy(*handle, m * n, Q_, 1, A, 1);
  #else
    res = cublasDcopy(*handle, m * n, Q_, 1, A, 1);
  #endif
  CHECK_CUBLAS_RETURN(res, "Failed to copy A = Q'")

  return 0;
}


// Wrapper kernel to house_qr_q. Use only for testing.
__global__ void TileQR_house_qr_q_kernel(Numeric *Y,
                                         Numeric *T,
                                         Numeric *Q,
                                         Numeric *Q_,
                                         int m, int n)
{
    cublasHandle_t handle;
    int res = cublasCreate(&handle);

    house_qr_q(&handle, Y, T, Q, Q_, m, n);
}

// Wrapper function to single-threaded cublasDgemm_hmn kernel. Use only for testing.
extern "C"
int
TileQR_house_qr_q(Matrix *Y,
                  Matrix *T,
                  Matrix *Q,
                  Matrix *Q_,
                  int m, int n)
{    
    TileQR_house_qr_q_kernel<<<1, 1>>>(Y->data_d, T->data_d, Q->data_d, Q_->data_d, m, n);
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