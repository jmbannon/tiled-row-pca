#include "../TileQR_Operations.h"
#include "../constants.h"
#include "../error.h"
#include "../Vector.h"
#include "../Matrix.h"
#include "../BlockMatrix.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdbool.h>

// cublasStrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, diag, m, k, &alpha, B, ldb, A, lda, C, ldc);
__device__ void trmm1(const Numeric alpha,
                      const Numeric *A, int lda,
                      const Numeric *B, int ldb,
                      Numeric *C, int ldc) {
  int c_idx;

  #pragma unroll
  for (int i = 0; i < BLK_LEN; i++) {

    #pragma unroll
    for (int j = 0; j < BLK_LEN; j++) {

      c_idx = MAT_POS(i, j, ldc);
      C[c_idx] = A[MAT_POS(i, j, lda)];

      for (int k = 0; k < j; k++) {
          C[c_idx] += A[MAT_POS(i, k, lda)] * B[MAT_POS(j, k, ldb)];
      }

      C[c_idx] *= alpha;
    }
  }
}

// res = cublasStrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, diag, k, n, &alpha, A, lda, B, ldb, C, ldc);
__device__ void trmm2(const Numeric alpha,
                      const Numeric *A, int lda,
                      const Numeric *B, int ldb,
                      Numeric *C, int ldc) {
  int c_idx;

  #pragma unroll
  for (int i = 0; i < BLK_LEN; i++) {

    #pragma unroll
    for (int j = 0; j < BLK_LEN; j++) {

      c_idx = MAT_POS(i, j, ldc);
      C[c_idx] = B[MAT_POS(i, j, ldb)];

      for (int k = 0; k < i; k++) {
          C[c_idx] += A[MAT_POS(i, k, lda)] * B[MAT_POS(k, j, ldb)];
      }

      C[c_idx] *= alpha;
    }
  }
}


__device__ void norm(int n, const Numeric *x, Numeric *result) {
  *result = 0;
  for (int i = 0; i < n; i++) {
    *result += (x[i] * x[i]);
  }
  #if FLOAT_NUMERIC
    *result = sqrtf(*result);
  #else
    *result = sqrt(*result);
  #endif
}

__device__ void geam(const Numeric alpha,
                     const Numeric *A, int lda,
                     const Numeric beta,
                     const Numeric *B, int ldb,
                     Numeric *C, int ldc) {
  #pragma unroll
  for (int i = 0; i < BLK_LEN; i++) {

    #pragma unroll
    for (int j = 0; j < BLK_LEN; j++) {
      C[MAT_POS(i, j, ldc)] = (alpha * A[MAT_POS(i, j, lda)]) + (beta * B[MAT_POS(i, j, ldb)]);
    }
  }

}

__device__ void gemm(const Numeric alpha,
                     const Numeric *A, int lda,
                     const Numeric *B, int ldb,
                     const Numeric beta,
                     Numeric *C, int ldc) {
  int c_idx;
  #pragma unroll
  for (int i = 0; i < BLK_LEN; i++) {

    #pragma unroll
    for (int j = 0; j < BLK_LEN; j++) {

      c_idx = MAT_POS(i, j, ldc);
      C[c_idx] = (beta == 0.0) ? 0.0 : C[c_idx] * beta;

      #pragma unroll
      for (int k = 0; k < BLK_LEN; k++) {
          C[c_idx] += A[MAT_POS(i, k, lda)] * B[MAT_POS(k, j, ldb)];
      }
      C[c_idx] *= alpha;
    }
  }
}

__device__ void gemtm(const Numeric alpha,
                      const Numeric *A, int lda,
                      const Numeric *B, int ldb,
                      const Numeric beta,
                      Numeric *C, int ldc) {
  int c_idx;
  #pragma unroll
  for (int i = 0; i < BLK_LEN; i++) {

    #pragma unroll
    for (int j = 0; j < BLK_LEN; j++) {

      c_idx = MAT_POS(i, j, ldc);
      C[c_idx] = (beta == 0.0) ? 0.0 : C[c_idx] * beta;

      #pragma unroll
      for (int k = 0; k < BLK_LEN; k++) {
          C[c_idx] += A[MAT_POS(k, i, lda)] * B[MAT_POS(k, j, ldb)];
      }
      C[c_idx] *= alpha;
    }
  }
}

__device__ void gemtmt(const Numeric alpha,
                       const Numeric *A, int lda,
                       const Numeric *B, int ldb,
                       const Numeric beta,
                       Numeric *C, int ldc) {
  int c_idx;

  #pragma unroll
  for (int i = 0; i < BLK_LEN; i++) {

    #pragma unroll
    for (int j = 0; j < BLK_LEN; j++) {

      c_idx = MAT_POS(i, j, ldc);
      C[c_idx] = (beta == 0.0) ? 0.0 : C[c_idx] * beta;

      #pragma unroll
      for (int k = 0; k < BLK_LEN; k++) {
          C[c_idx] += A[MAT_POS(k, i, lda)] * B[MAT_POS(j, k, ldb)];
      }
      C[c_idx] *= alpha;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// DGEQT2
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
  * Given an n-vector x, computes an n-vector v with v[0] = 1 such that (I - 2*v*t(v) / t(v) * v)x is 
  * zero in all but the first component.
  * @param x n-vector used to compute v.
  * @param v n-vector to store output
  * @param n Length of vectors x and v
  *
  * @see{https://www.youtube.com/watch?v=d-yPM-bxREs}
  */
__device__ int house(cublasHandle_t *handle, Numeric *x, Numeric *v, int n)
{
    Numeric x_norm;

    cudaDeviceSynchronize();
    for (int i = 0; i < n; i++) {
      v[i] = x[i];
    }

    norm(n, x, &x_norm);

    if (x_norm != 0) {
    	const Numeric sign = x[0] >= 0 ? 1.0 : -1.0;
    	const Numeric beta = 1.0 / (x[0] + (sign * x_norm));

      for (int i = 1; i < n; i++) {
        v[i] *= beta;
      }
    }
    v[0] = 1.0;

    return 0;
}

/**
  * Overwrites A with PA where P = (I - 2*v*t(v) / t(v) * v).
  * The following algorithm is from Golub, Van Loan Matrix Computations:
  *
  * function: A = row.house(A, v)
  *     beta = -2 / (t(v) * v)
  *     w = beta * t(A) * v
  *     A = A + v * t(w)
  *
  * @param A m-by-n matrix
  * @param v m-vector v with v[0] = 1.0 {@see house}
  * @param beta Scalar used in the transformation.
  * @param w n-vector temporary storage
  */
__device__ int house_row(cublasHandle_t *handle, Numeric *A, Numeric *v, Numeric *beta, Numeric *w, int m, int n, int ldm)
{
    int res = 0;

    norm(m, v, beta);
    CHECK_CUBLAS_RETURN(res, "Failed to compute beta");

    *beta = -2.0 / (*beta * *beta);

    // Computes w
    Numeric w_scalar = 0.0;
    #if FLOAT_NUMERIC
      res = cublasSgemv(*handle, CUBLAS_OP_T, m, n, beta, A, ldm, v, 1, &w_scalar, w, 1);
    #else
      res = cublasDgemv(*handle, CUBLAS_OP_T, m, n, beta, A, ldm, v, 1, &w_scalar, w, 1);
    #endif
    CHECK_CUBLAS_RETURN(res, "Failed to compute w");

    // Annihilate column of A: A = A + v * t(w)
    Numeric scalar = 1.0;
    #if FLOAT_NUMERIC
      res = cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, 1, &scalar, v, m, w, 1, &scalar, A, ldm);
    #else
      res = cublasDgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, 1, &scalar, v, m, w, 1, &scalar, A, ldm);
    #endif
    CHECK_CUBLAS_RETURN(res, "Failed to annihilate column in A");

    return res;
}

/**
  * Produces an upper triangular matrix R, unit lower triangular matrix V that contains n Householder reflectors.
  * R and V are written on the memory area used for A.
  *
  * @param A upper triangular matrix R. Lower triangular contains partial householder reflectors:
  *        all diagonal elements should be 1 to represent full householder reflector.
  * @param w Work-space vector.
  * @param store_house True if householder vectors should be stored in lower-triangular portion of output. False otherwise.
  */
__device__ int house_qr(cublasHandle_t *handle, Numeric *A, Numeric *beta, Numeric *w, bool store_house, int m, int n)
{
  int res;
  Numeric *v;

  res = cudaMalloc(&v, m * sizeof(Numeric));
  CHECK_SUCCESS_RETURN(res);

  for (int j = 0; j < n; j++) {
    int pos = MAT_POS(j, j, m);
    res = house(handle, &A[pos], &v[j], m - j);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to compute house");

    res = house_row(handle, &A[pos], &v[j], &beta[j], &w[j], m - j, n - j, m);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to compute house_row");

    // Copies householder vector into lower triangular portion of A
    if (store_house && j < m) {
      for (int i = 1; i < m - j; i++) {
        A[pos + i] = v[j + i];
      }
    }
  }

  res = cudaFree(v);
  CHECK_SUCCESS_RETURN(res);

  return res;
}


/**
  * Computes the Matrix T such that P_1 * ... * P_n = I + (Y * T * t(Y))
  * where P_i are the householder matrices that upper triangularize A (i.e. R in QR) during the ith step.
  * 
  * For j = 1:r
  *   if j == 1 then
  *     Y = [v_1]; T = [-2]
  *   else
  *     z = -2 * T * t(Y) * v_j
  *     Y = [Y v_j]
  *
  *     T = [T  z]
  *         [0 -2]
  *   endif
  * end j
  *
  * Note it assumes householder vectors in Y are normalized. We change the algorithm slightly to use 
  * previously calculated betas (-2 / t(v) * v == -2 / ||v||^2). We replace '-2' with beta[j].
  *
  * @see{https://www.cs.cornell.edu/cv/ResearchPDF/A%20Storage-Efficient%20WY%20Representation%20for%20Products%20of%20Householder%20Transformations.pdf}
  *
  * @param Y m-by-n matrix where lower-triangular + diag portion holds householder vectors and the upper-triangular portion holds the R matrix from QR.
  * @param T n-by-n output matrix to store T
  */
__device__ int house_yt(cublasHandle_t *handle, Numeric *Y, Numeric *T, Numeric *beta, int m, int n)
{
  int res;
  Numeric alpha;
  Numeric zero = 0.0;
  int v_idx;
  int z_idx;
  int y_idx;

  T[0] = beta[0];
  for (int j = 1; j < n; j++) {
    alpha = beta[j];

    y_idx = MAT_POS(j, 0, m);
    v_idx = MAT_POS(j, j, m);
    z_idx = MAT_POS(0, j, n);

    // Computes -2 * t(Y) * v_j = z' in an optimized way to ignore 0 elements in v_j. Stores it in z-location of T matrix.
    #if FLOAT_NUMERIC
      res = cublasSgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_N, j, 1, m - j, &alpha, &Y[y_idx], m, &Y[v_idx], m, &zero, &T[z_idx], n);
    #else
      res = cublasDgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_N, j, 1, m - j, &alpha, &Y[y_idx], m, &Y[v_idx], m, &zero, &T[z_idx], n);
    #endif
    CHECK_CUBLAS_RETURN(res, "Failed to compute -2 * t(Y) * v_j matrix-matrix multiplication in house_yt");

    // Computes T * z' using a triangular matrix-vector multiplication routine.
    #if FLOAT_NUMERIC
      res = cublasStrmv(*handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, j, T, n, &T[z_idx], 1);
    #else
      res = cublasDtrmv(*handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, j, T, n, &T[z_idx], 1);
    #endif
    CHECK_CUBLAS_RETURN(res, "Failed to compute T * z' triangular matrix-vector multiplication in house_yt");

    T[MAT_POS(j, j, n)] = beta[j];
  }
  return 0;
}

/**
  * Performs QR decomposition on a m-by-n matrix A. Computes upper-triangular matrix R,
  * where A = QR and Householder vectors Y are stored in lower-diagonal portion of R.
  * Uses Y to compute T, where Q = I + Y %*% T %*% t(Y).
  *
  * @param A m-by-n matrix.
  * @param T n-by-n output matrix.
  * @return R, Y, T, where A = QR and T for Q = I + Y %*% T %*% t(Y). Overwrites A with R and Y.
  */
__device__ int dgeqt2(cublasHandle_t *handle, Numeric *A, Numeric *T, int m, int n)
{
  int res;
  // Temporary work matrix
  Numeric *w;
  Numeric *beta;

  res = cudaMalloc(&w, m * sizeof(Numeric));
  CHECK_SUCCESS_RETURN(res);

  res = cudaMalloc(&beta, n * sizeof(Numeric));
  CHECK_SUCCESS_RETURN(res);

  res = house_qr(handle, A, beta, w, true, m, n);
  CHECK_ZERO_ERROR_RETURN(res, "Failed to compute house_qr in dgeqt2");

  // Restore householder vectors for YT Generation. Store diag in work vector.
  int diag_idx;
  for (int i = 0; i < n; i++) {
    diag_idx = MAT_POS(i, i, m);
    w[i] = A[diag_idx];
    A[diag_idx] = 1.0;

  }

  res = house_yt(handle, A, T, beta, m, n);
  CHECK_ZERO_ERROR_RETURN(res, "Failed to compute house_yt in dgeqt2");

  for (int i = 0; i < n; i++) {
    diag_idx = MAT_POS(i, i, m);
    A[diag_idx] = w[i];
  }

  res = cudaFree(w);
  CHECK_SUCCESS_RETURN(res);

  res = cudaFree(beta);
  CHECK_SUCCESS_RETURN(res);

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// DTSQT2
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
  * Performs DGEQT2 on a row-binded matrix rbind(R, A).
  *
  * @param handle cuBLAS handle
  * @param R n-by-n upper-triangular matrix to row-bind. May have non-zero elements in lower diagonal. Stores output R matrix from DGEQT2 here.
  * @param A n-by-n matrix to row-bind. Stores lower portion of householder vectors from DGEQT2 in here. The 'hessianberg' portion of the householder
  *          vectors is an identity matrix, so there is no need to store that.
  * @param T n-by-n output matrix.
  * @param RA_rowbind 2n-by-n work matrix to store the row-bind and compute DGEQT2 on.
  * @param zero_tri True if lower-triangular portion of R needs to be zeroed. False otherwise.
  */
__device__ int dtsqt2(cublasHandle_t *handle, Numeric *R, Numeric *A, Numeric *T, Numeric *RA_rowbind, bool zero_tri, int n)
{
  int res;
  int RArows = 2*n;

  // TODO: Optimize

  // Stores R into upper-portion of RA_rowbind. Zeroes lower-triangular portion.
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++) {
      RA_rowbind[MAT_POS(i, j, RArows)] = (i > j) ? 0.0 : R[MAT_POS(i, j, n)];
    }
  }

  // Stores A into lower-portion of RA_rowbind.
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++) {
      RA_rowbind[MAT_POS(n + i, j, RArows)] = A[MAT_POS(i, j, n)];
    }
  }

  res = dgeqt2(handle, RA_rowbind, T, RArows, n);
  CHECK_ZERO_ERROR_RETURN(res, "Failed to compute dgeqt2 on row-binded matrix");

  // Stores output R matrix into upper-triangular portion of R
  for (int j = 0; j < n; j++) {
    for (int i = 0; i <= j; i++) {
      R[MAT_POS(i, j, n)] = RA_rowbind[MAT_POS(i, j, RArows)];
    }
  }

  // Stores output householder vectors into A.
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++) {
      A[MAT_POS(i, j, n)] = RA_rowbind[MAT_POS(n + i, j, RArows)];
    }
  }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// DSSRFB
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
  * Performs DSSRFB
  * The original operation is rbind(A_kj, A_ij) = t(I + (V * T * t(V))) * rbind(A_kj, A_ij)
  * It can be optimized to the following:
  * Let X = t(T) * A_kj
  * Let Y = t(T) * t(V) * A_ij
  * Let Z = X + Y
  *
  * rbind(A_kj, A_ij) = rbind(A_kj + Z, A_ij + (V * Z))
  *
  *
  *
  */
__device__ int dssrfb(Numeric *A_kj,
                      Numeric *A_ij,
                      Numeric *V,
                      Numeric *T,
                      Numeric *X, int ldx,
                      Numeric *Y, int ldy,
                      int n)
{
  const Numeric alpha = 1.0;
  const Numeric zero = 0.0;

  // X = t(T) * A_kj
  gemtm(alpha, T, n, A_kj, n, zero, X, ldx);

  // Y' = t(T) * t(V)
  gemtmt(alpha, T, n, V, n, zero, Y, ldy);

  // Z = X = Y' * A_ij + X
  gemm(alpha, Y, ldy, A_ij, n, alpha, X, ldx);

  // A_kj = A_kj + Z
  geam(alpha, A_kj, n, alpha, X, ldx, A_kj, n);

  // A_ij = (V * Z) + A_ij
  gemm(alpha, V, n, X, ldx, alpha, A_ij, n);

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// DLAFRB
////////////////////////////////////////////////////////////////////////////////////////////////////


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
__device__ int house_qr_q(Numeric *Y, Numeric *T, Numeric *Q, Numeric *Q_)
{
    // int res;
    Numeric alpha = 1.0;
    
    // Calculates Q' = Y * T
    trmm2(alpha, Y, BLK_LEN, T, BLK_LEN, Q_, BLK_LEN);

    // Calculates Q = Q' * t(Y)
    //              = Y * T * t(Y)
    trmm1(alpha, Q_, BLK_LEN, Y, BLK_LEN, Q, BLK_LEN);

    // Calculates Q = I + Q
    //              = I + (Y * T * t(Y))
    #pragma unroll
    for (int i = 0; i < BLK_LEN; i++) {
      Q[MAT_POS(i, i, BLK_LEN)] += 1.0;
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
__device__ int dlarfb(cublasHandle_t *handle, Numeric *A, Numeric *Y, Numeric *T, Numeric *Q, Numeric *Q_)
{
  int res;

  res = house_qr_q(Y, T, Q, Q_);
  CHECK_ZERO_ERROR_RETURN(res, "Failed to compute house_qr_q");

  const Numeric zero = 0.0;
  const Numeric alpha = 1.0;

  gemtm(alpha, Q, BLK_LEN, A, BLK_LEN, zero, Q_, BLK_LEN);

  #pragma unroll
  for (int i = 0; i < BLK_SIZE; i++) {
    A[i] = Q_[i];
  }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// TileQR
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ int BlockMatrix_TileQR_single_thread_kernel(Numeric *A, int blk_m, int blk_n)
{
  int res;
  Numeric *T;
  Numeric *Rbind;
  Numeric *Q;
  Numeric *Q_;

  int min_blk_d = blk_m > blk_n ? blk_n : blk_m;

  cublasHandle_t handle;
  res = cublasCreate(&handle);
  CHECK_CUBLAS_RETURN(res, "Failed to init handle");

  res = cudaMalloc(&T, BLK_SIZE_MEM);
  CHECK_CUBLAS_RETURN(res, "Failed to init T");

  res = cudaMalloc(&Rbind, 2 * BLK_SIZE_MEM);
  CHECK_CUBLAS_RETURN(res, "Failed to init Rbind");

  res = cudaMalloc(&Q, BLK_SIZE_MEM);
  CHECK_CUBLAS_RETURN(res, "Failed to init Q");

  res = cudaMalloc(&Q_, BLK_SIZE_MEM);
  CHECK_CUBLAS_RETURN(res, "Failed to init Q'");

  for (int i = 0; i < BLK_SIZE; i++) {
    T[i] = 0;
    Q[i] = 0;
    Q_[i] = 0;
    Rbind[i] = 0;    
  }

  for (int i = BLK_SIZE; i < 2*BLK_SIZE; i++) {
    Rbind[i] = 0;
  }

  for (int k = 0; k < min_blk_d; k++) {
    Numeric *A_kk = &A[BLK_POS(k, k, blk_n)];

    res = dgeqt2(&handle, A_kk, T, BLK_LEN, BLK_LEN);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to compute dgeqt2");

    for (int n = (k + 1); n < blk_n; n++) {

      Numeric *A_kn = &A[BLK_POS(k, n, blk_n)];

      res = dlarfb(&handle, A_kn, A_kk, T, Q, Q_);
      CHECK_ZERO_ERROR_RETURN(res, "Failed to compute dlarfb");
    }

    for (int m = (k + 1); m < blk_m; m++) {

      Numeric *A_mk = &A[BLK_POS(m, k, blk_n)];
      for (int i = 0; i < BLK_SIZE; i++) {
        T[i] = 0;  
      }

      res = dtsqt2(&handle, A_kk, A_mk, T, Rbind, true, BLK_LEN);
      CHECK_ZERO_ERROR_RETURN(res, "Failed to compute dtsqt2");

      for (int n = (k + 1); n < blk_n; n++) {
        Numeric *A_kn = &A[BLK_POS(k, n, blk_n)];
        Numeric *A_mn = &A[BLK_POS(m, n, blk_n)];

        res = dssrfb(A_kn, A_mn, A_mk, T, Rbind, DBL_BLK_LEN, &Rbind[BLK_LEN], DBL_BLK_LEN, BLK_LEN);
        CHECK_ZERO_ERROR_RETURN(res, "Failed to compute dssrfb");

        cudaDeviceSynchronize();
      }
    }
  }

  res = cudaFree(T);
  CHECK_SUCCESS_RETURN(res);

  res = cudaFree(Rbind);
  CHECK_SUCCESS_RETURN(res);

  res = cudaFree(Q);
  CHECK_SUCCESS_RETURN(res);

  res = cudaFree(Q_);
  CHECK_SUCCESS_RETURN(res);

  return 0;
}

//////////////////////
// Parallel QR
//////////////////////

/**
  * Multiplies a matrix A s.t. A = t(Q) * A
  *                              = t(I + (Y * T * t(Y))) * A
  * where Q is from a diagonal tile P, where P = QR, and A is an adjacent tile to the right of P.
  *
  * @param M Pointer to beginning of BlockMatrix M
  * @param i Block row to perform on
  * @param T n-by-n matrix
  *
  */

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
  *
  * res = dlarfb(&handle, A_kn, A_kk, T, Q, Q_, BLK_LEN, BLK_LEN);
  */
__global__ void dlarfb_kernel(Numeric *M, int lbdm, int k, Numeric *T) {
    Numeric *Q;
    Numeric *Q_;
    cublasHandle_t handle;
    int res = cublasCreate(&handle);
    if (res != 0) {
      printf("failed!!!\n");
    }
    // check res

    res = cudaMalloc(&Q, BLK_SIZE_MEM);
    if (res != 0) {
      printf("failed!!!\n");
    }
    // check res

    res = cudaMalloc(&Q_, BLK_SIZE_MEM);
    if (res != 0) {
      printf("failed!!!\n");
    }
    //check res

    Numeric *M_kk = &M[BLK_POS(k, k, lbdm)];
    Numeric *M_kn = &M[BLK_POS(k, k + 1 + threadIdx.x, lbdm)];

    res = dlarfb(&handle, M_kn, M_kk, T, Q, Q_);
    // check res

    cudaFree(Q);
    cudaFree(Q_);

    cublasDestroy(handle);
}


/**
 * Performs DGEQT2 on a diagonal block of the BlockMatrix A.
 *
 * @param A Pointer to start of A
 * @param ldba Leading block dimension of A
 * @param T BLK_LEN-by-BLK_LEN storage matrix for result
 * @param k Block row and column (diagonal block) to perform DGEQT2 on
 */
__global__ void dgeqt2_dlarfb_row_kernel(Numeric *M, int lbdm, int k, int nr_blk_rows, int nr_blk_cols) {
    Numeric *T;
    Numeric *M_kk;
    cublasHandle_t handle;

    int res = cublasCreate(&handle);
    // check res

    res = cudaMalloc(&T, BLK_SIZE_MEM);
    for (int i = 0; i < BLK_SIZE; i++) {
        T[i] = 0;
    }
    // check res

    M_kk = &M[BLK_POS(k, k, lbdm)];

    res = dgeqt2(&handle, M_kk, T, BLK_LEN, BLK_LEN);
    // check res

    if (k < nr_blk_cols - 1) {
      dlarfb_kernel<<<1, nr_blk_cols - k - 1>>>(M, lbdm, k, T);
    }

    cudaDeviceSynchronize();

    cudaFree(T);
    cublasDestroy(handle);
}

__global__ void dssrfb_kernel(Numeric *M, int lbdm,
                              int k, int m,
                              Numeric *V,
                              Numeric *T) {
  Numeric X[BLK_SIZE];
  Numeric Y[BLK_SIZE];

  Numeric *A_kn = &M[BLK_POS(k, k + 1 + threadIdx.x, lbdm)];
  Numeric *A_mn = &M[BLK_POS(m, k + 1 + threadIdx.x, lbdm)];

  dssrfb(A_kn, A_mn, V, T, X, BLK_LEN, Y, BLK_LEN, BLK_LEN);
}

__global__ void dtsqt2_dssrfb_row_kernel(Numeric *M, int lbdm, int k, int m, int nr_blk_cols) {
    Numeric *T;
    Numeric *Rbind;
    cublasHandle_t handle;

    int res = cublasCreate(&handle);
    // check res

    res = cudaMalloc(&T, BLK_SIZE_MEM);
    // check res

    res = cudaMalloc(&Rbind, 2 * BLK_SIZE_MEM);
    //check res

    for (int i = 0; i < BLK_SIZE; i++) {
        T[i] = 0;
        Rbind[i] = 0;    
    }

    for (int i = BLK_SIZE; i < 2*BLK_SIZE; i++) {
        Rbind[i] = 0;
    }

    Numeric *A_mk = &M[BLK_POS(m, k, lbdm)];
    Numeric *A_kk = &M[BLK_POS(k, k, lbdm)];

    res = dtsqt2(&handle, A_kk, A_mk, T, Rbind, true, BLK_LEN);
    // check res

    cudaDeviceSynchronize();

    if (k != nr_blk_cols - 1) {
      dssrfb_kernel<<<1, nr_blk_cols - k - 1>>>(M, lbdm, k, m, A_mk, T);
    }

    cudaDeviceSynchronize();
    cudaFree(T);
    cudaFree(Rbind);
    cublasDestroy(handle);
}

// extern "C"
// int
// BlockMatrix_TileQR_multi_thread(BlockMatrix *BlkM)
// {    
//   int res = 0;
//   Numeric *M = BlkM->data_d;

//   int blk_m = BlkM->nr_blk_rows;
//   int blk_n = BlkM->nr_blk_cols;

//   int min_blk_d = blk_m > blk_n ? blk_n : blk_m;

//   for (int k = 0; k < min_blk_d; k++) {
//     dgeqt2_dlarfb_row_kernel<<<1,1>>>(M, blk_n, k, blk_m, blk_n); // check res
//     CHECK_ZERO_ERROR_RETURN(res, "Failed to compute dgeqt2");

//     cudaDeviceSynchronize();
//     for (int m = (k + 1); m < blk_m; m++) {
//       dtsqt2_dssrfb_row_kernel<<<1, 1>>>(M, blk_n, k, m, blk_n); // check res
//       CHECK_ZERO_ERROR_RETURN(res, "Failed to compute row kernel");
//       cudaDeviceSynchronize();
//     }
//   }

//   cudaDeviceSynchronize();
//   return 0;
// }

extern "C"
int
BlockMatrix_TileQR_multi_thread(BlockMatrix *BlkM)
{    
  bool dgeqt2_running = true;
  bool dlarfb_running = true;
  bool ran_dgeqt2 = false;

  int blk_m = BlkM->nr_blk_rows;
  int blk_n = BlkM->nr_blk_cols;

  int min_blk_d = blk_m > blk_n ? blk_n : blk_m;
  Numeric *M = BlkM->data_d;
  
  int i = 0;
  int j;
  int j_start = 0;
  int *dl = (int *)calloc(min_blk_d, sizeof(int));
  
  dgeqt2_dlarfb_row_kernel<<<1,1>>>(M, blk_n, 0, blk_m, blk_n);
  dl[i] = i + 1;
  ++i;

  cudaDeviceSynchronize();
  
  while (dgeqt2_running || dlarfb_running) {
    if (i < min_blk_d && i < dl[i - 1]) {
      dgeqt2_dlarfb_row_kernel<<<1,1>>>(M, blk_n, i, blk_m, blk_n); // check res
      ran_dgeqt2 = true;
    } else if (i == min_blk_d) {
      dgeqt2_running = false;
    }

    j = j_start;
    while (dl[j] != 0 && j < min_blk_d) {
      if (dl[j] < blk_m) {
        dtsqt2_dssrfb_row_kernel<<<1, 1>>>(M, blk_n, j, dl[j], blk_n);
        dl[j] += 1;
      } else if (dl[j] == blk_m) {
        j_start = j + 1;
      }
      ++j;
    }

    if (j_start == min_blk_d) {
      dlarfb_running = false;
    }

    if (ran_dgeqt2) {
      dl[i] = i + 1;
      ++i;
      ran_dgeqt2 = false;
    }

    cudaDeviceSynchronize();
  }

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// KERNEL WRAPPERS
////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void TileQR_wrapper(Numeric *A, int blk_m, int blk_n)
{
  BlockMatrix_TileQR_single_thread_kernel(A, blk_m, blk_n);
}

extern "C"
int
BlockMatrix_TileQR_single_thread(BlockMatrix *A)
{
  TileQR_wrapper<<<1, 1>>>(A->data_d, A->nr_blk_rows, A->nr_blk_cols);
  cudaDeviceSynchronize();
  return 0;
}


__global__ void house_kernel(Numeric *x, Numeric *v, int n) {
    cublasHandle_t handle;
    int res = cublasCreate(&handle);
    house(&handle, x, v, n);
}

extern "C"
int
TileQR_house(cublasHandle_t *handle, Vector *in, Vector *out) {    
    house_kernel<<<1, 1>>>(in->data_d, out->data_d, in->nr_elems);
    return 0;
}

__global__ void dgeqt2_kernel_test(Numeric *A, Numeric *T, int m, int n) {
    cublasHandle_t handle;
    int res = cublasCreate(&handle);

    dgeqt2(&handle, A, T, m, n);
}

extern "C"
int
TileQR_dgeqt2(cublasHandle_t *handle, Matrix *A, Matrix *T) {    
    dgeqt2_kernel_test<<<1, 1>>>(A->data_d, T->data_d, A->nr_rows, A->nr_cols);
    return 0;
}

// Wrapper kernel to house_qr_q. Use only for testing.
__global__ void TileQR_house_qr_q_kernel(Numeric *Y,
                                         Numeric *T,
                                         Numeric *Q,
                                         Numeric *Q_,
                                         int m, int n)
{
    house_qr_q(Y, T, Q, Q_);
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
