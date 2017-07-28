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

/**
  * Computes C = A * t(B) where B is a lower-filled triangular matrix with ones on its diagonal.
  */
__device__ void blk_trmmr(const Numeric alpha,
                          const Numeric *A,
                          const Numeric *B,
                          Numeric *C) {
  register int c_idx;

  #pragma unroll
  for (int i = 0; i < BLK_LEN; i++) {

    #pragma unroll
    for (int j = 0; j < BLK_LEN; j++) {

      c_idx = MAT_POS(i, j, BLK_LEN);
      C[c_idx] = A[MAT_POS(i, j, BLK_LEN)];

      for (int k = 0; k < j; k++) {
          C[c_idx] += A[MAT_POS(i, k, BLK_LEN)] * B[MAT_POS(j, k, BLK_LEN)];
      }

      C[c_idx] *= alpha;
    }
  }
}

/**
  * Computes C = A * B where A is a lower-filled triangular matrix with ones on its diagonal.
  */
__device__ void blk_trmml(const Numeric alpha,
                          const Numeric *A,
                          const Numeric *B,
                          Numeric *C) {
  register int c_idx;

  #pragma unroll
  for (int i = 0; i < BLK_LEN; i++) {

    #pragma unroll
    for (int j = 0; j < BLK_LEN; j++) {

      c_idx = MAT_POS(i, j, BLK_LEN);
      C[c_idx] = B[MAT_POS(i, j, BLK_LEN)];

      for (int k = 0; k < i; k++) {
          C[c_idx] += A[MAT_POS(i, k, BLK_LEN)] * B[MAT_POS(k, j, BLK_LEN)];
      }

      C[c_idx] *= alpha;
    }
  }
}

//cublasStrmv(*handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, j, T, n, &T[z_idx], 1);
__device__ void trmv(int m,
                     const Numeric *A, int lda,
                     Numeric *C) {
  for (int i = 0; i < m; i++) {
    C[i] *= A[MAT_POS(i, i, lda)];
    for (int j = i + 1; j < m; j++) {
      C[i] += A[MAT_POS(i, j, lda)] * C[j];
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

__device__ void gemtm(int m, int n, int p,
                      const Numeric alpha,
                      const Numeric *A, int lda,
                      const Numeric *B, int ldb,
                      const Numeric beta,
                      Numeric *C, int ldc) {
  int c_idx;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {

      c_idx = MAT_POS(i, j, ldc);
      C[c_idx] = (beta == 0.0) ? 0.0 : C[c_idx] * beta;

      for (int k = 0; k < p; k++) {
          C[c_idx] += A[MAT_POS(k, i, lda)] * B[MAT_POS(k, j, ldb)];
      }
      C[c_idx] *= alpha;
    }
  }
}

__device__ void gemm(int m, int n, int p,
                     const Numeric alpha,
                     const Numeric *A, int lda,
                     const Numeric *B, int ldb,
                     const Numeric beta,
                     Numeric *C, int ldc) {
  int c_idx;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {

      c_idx = MAT_POS(i, j, ldc);
      C[c_idx] = (beta == 0.0) ? 0.0 : C[c_idx] * beta;

      for (int k = 0; k < p; k++) {
          C[c_idx] += A[MAT_POS(i, k, lda)] * B[MAT_POS(k, j, ldb)];
      }
      C[c_idx] *= alpha;
    }
  }
}

/**
  * Computes c = alpha * t(A)b
  */
__device__ void gemtv(int m, int n,
                      const Numeric alpha,
                      const Numeric *A, int lda,
                      const Numeric *b,
                      Numeric *C) {
  // int c_idx;
  for (int j = 0; j < n; j++) {
    C[j] = 0;
    for (int i = 0; i < m; i++) {
      C[j] += A[MAT_POS(i, j, lda)] * b[i];
    }
    C[j] *= alpha;
  }
}

__device__ void blk_gemm(const Numeric alpha,
                         const Numeric *A,
                         const Numeric *B,
                         const Numeric beta,
                         Numeric *C) {
  int c_idx;
  #pragma unroll
  for (int i = 0; i < BLK_LEN; i++) {

    #pragma unroll
    for (int j = 0; j < BLK_LEN; j++) {

      c_idx = MAT_POS(i, j, BLK_LEN);
      C[c_idx] = (beta == 0.0) ? 0.0 : C[c_idx] * beta;

      #pragma unroll
      for (int k = 0; k < BLK_LEN; k++) {
          C[c_idx] += A[MAT_POS(i, k, BLK_LEN)] * B[MAT_POS(k, j, BLK_LEN)];
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
__device__ int house(Numeric *x, Numeric *v, int n)
{
    Numeric x_norm;
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
__device__ int house_row(Numeric *A, Numeric *v, Numeric *beta, Numeric *w, int m, int n, int ldm)
{
    const Numeric alpha = 1.0;
    norm(m, v, beta);

    *beta = -2.0 / (*beta * *beta);

    // Computes w
    gemtv(m, n, *beta, A, ldm, v, w);

    // Annihilate column of A: A = A + v * t(w)
    gemm(m, n, 1, alpha, v, m, w, 1, alpha, A, ldm);

    return 0;
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
__device__ int house_qr(Numeric *A, Numeric *beta, Numeric *w, bool store_house, int m, int n)
{
  int res;
  Numeric *v;

  res = cudaMalloc(&v, m * sizeof(Numeric));
  CHECK_SUCCESS_RETURN(res);

  for (int j = 0; j < n; j++) {
    int pos = MAT_POS(j, j, m);

    house(&A[pos], &v[j], m - j);
    house_row(&A[pos], &v[j], &beta[j], &w[j], m - j, n - j, m);

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
__device__ int house_yt(Numeric *Y, Numeric *T, Numeric *beta, int m, int n)
{
  // int res;
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
    gemtm(j, 1, m - j, alpha, &Y[y_idx], m, &Y[v_idx], m, zero, &T[z_idx], n);

    // Computes T * z' using a triangular matrix-vector multiplication routine.
    trmv(j, T, n, &T[z_idx]);

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

  house_qr(A, beta, w, true, m, n);

  // Restore householder vectors for YT Generation. Store diag in work vector.
  int diag_idx;
  for (int i = 0; i < n; i++) {
    diag_idx = MAT_POS(i, i, m);
    w[i] = A[diag_idx];
    A[diag_idx] = 1.0;

  }

  house_yt(A, T, beta, m, n);

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

/**
  * Performs QR decomposition on a m-by-n matrix A. Computes upper-triangular matrix R,
  * where A = QR and Householder vectors Y are stored in lower-diagonal portion of R.
  * Uses Y to compute T, where Q = I + Y %*% T %*% t(Y).
  *
  * @param A m-by-n matrix.
  * @param T n-by-n output matrix.
  * @return R, Y, T, where A = QR and T for Q = I + Y %*% T %*% t(Y). Overwrites A with R and Y.
  */
__device__ int dblk_dgeqt2(Numeric *A, Numeric *T)
{
  Numeric w[DBL_BLK_LEN];
  Numeric beta[BLK_LEN];

  house_qr(A, beta, w, true, DBL_BLK_LEN, BLK_LEN);

  // Restore householder vectors for YT Generation. Store diag in work vector.
  int diag_idx;
  for (int i = 0; i < BLK_LEN; i++) {
    diag_idx = MAT_POS(i, i, DBL_BLK_LEN);
    w[i] = A[diag_idx];
    A[diag_idx] = 1.0;
  }

  house_yt(A, T, beta, DBL_BLK_LEN, BLK_LEN);

  for (int i = 0; i < BLK_LEN; i++) {
    diag_idx = MAT_POS(i, i, DBL_BLK_LEN);
    A[diag_idx] = w[i];
  }

  return 0;
}

__device__ int blk_dgeqt2(Numeric *A, Numeric *T)
{
  Numeric w[BLK_LEN];
  Numeric beta[BLK_LEN];

  house_qr(A, beta, w, true, BLK_LEN, BLK_LEN);

  // Restore householder vectors for YT Generation. Store diag in work vector.
  int diag_idx;
  for (int i = 0; i < BLK_LEN; i++) {
    diag_idx = MAT_POS(i, i, BLK_LEN);
    w[i] = A[diag_idx];
    A[diag_idx] = 1.0;
  }

  house_yt(A, T, beta, BLK_LEN, BLK_LEN);

  for (int i = 0; i < BLK_LEN; i++) {
    diag_idx = MAT_POS(i, i, BLK_LEN);
    A[diag_idx] = w[i];
  }

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
  */
__device__ int dtsqt2(cublasHandle_t *handle, Numeric *R, Numeric *A, Numeric *T, Numeric *RA_rowbind)
{
  // Stores R into upper-portion of RA_rowbind. Zeroes lower-triangular portion.
  #pragma unroll
  for (int j = 0; j < BLK_LEN; j++) {
    #pragma unroll
    for (int i = 0; i < BLK_LEN; i++) {
      RA_rowbind[MAT_POS(i, j, DBL_BLK_LEN)] = (i > j) ? 0.0 : R[MAT_POS(i, j, BLK_LEN)];
    }
  }

  // Stores A into lower-portion of RA_rowbind.
  #pragma unroll
  for (int j = 0; j < BLK_LEN; j++) {
    #pragma unroll
    for (int i = 0; i < BLK_LEN; i++) {
      RA_rowbind[MAT_POS(BLK_LEN + i, j, DBL_BLK_LEN)] = A[MAT_POS(i, j, BLK_LEN)];
    }
  }

  dblk_dgeqt2(RA_rowbind, T);

  // Stores output R matrix into upper-triangular portion of R
  #pragma unroll
  for (int j = 0; j < BLK_LEN; j++) {
    for (int i = 0; i <= j; i++) {
      R[MAT_POS(i, j, BLK_LEN)] = RA_rowbind[MAT_POS(i, j, DBL_BLK_LEN)];
    }
  }

  // Stores output householder vectors into A.
  #pragma unroll
  for (int j = 0; j < BLK_LEN; j++) {
    #pragma unroll
    for (int i = 0; i < BLK_LEN; i++) {
      A[MAT_POS(i, j, BLK_LEN)] = RA_rowbind[MAT_POS(BLK_LEN + i, j, DBL_BLK_LEN)];
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
                      Numeric *X,
                      Numeric *Y)
{
  const Numeric alpha = 1.0;
  const Numeric zero = 0.0;

  // X = t(T) * A_kj
  gemtm(alpha, T, BLK_LEN, A_kj, BLK_LEN, zero, X, BLK_LEN);

  // Y' = t(T) * t(V)
  gemtmt(alpha, T, BLK_LEN, V, BLK_LEN, zero, Y, BLK_LEN);

  // Z = X = Y' * A_ij + X
  blk_gemm(alpha, Y, A_ij, alpha, X);

  // A_kj += Z
  #pragma unroll
  for (int i = 0; i < BLK_SIZE; i++) {
    A_kj[i] += X[i];
  }

  // A_ij = (V * Z) + A_ij
  blk_gemm(alpha, V, X, alpha, A_ij);

  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// DLAFRB
////////////////////////////////////////////////////////////////////////////////////////////////////

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
    blk_trmml(alpha, Y, T, Q_);

    // Calculates Q = Q' * t(Y)
    //              = Y * T * t(Y)
    blk_trmmr(alpha, Q_, Y, Q);

    // Calculates Q = I + Q
    //              = I + (Y * T * t(Y))
    #pragma unroll
    for (int i = 0; i < BLK_SIZE; i += BLK_LEN + 1) {
      Q[i] += 1.0;
    }

    return 0;
}


/**
  * Multiplies a matrix A s.t. A = t(Q) * A
  *                              = t(I + (Y * T * t(Y))) * A
  * where Q is from a diagonal tile P, where P = QR, and A is an adjacent tile to the right of P.
  *
  * @param A m-by-n matrix to multiply and override. Adjacent to the source tile of Q.
  * @param Y m-by-n Hessianberg matrix holding householder vectors.
  * @param T n-by-n matrix.
  * @param Q m-by-m matrix to store Q.
  * @param Q_ m-by-n work matrix.
  */
__device__ int dlarfb(Numeric *A, Numeric *Y, Numeric *T, Numeric *Q, Numeric *Q_)
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

    res = blk_dgeqt2(A_kk, T);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to compute dgeqt2");

    for (int n = (k + 1); n < blk_n; n++) {

      Numeric *A_kn = &A[BLK_POS(k, n, blk_n)];

      res = dlarfb(A_kn, A_kk, T, Q, Q_);
      CHECK_ZERO_ERROR_RETURN(res, "Failed to compute dlarfb");
    }

    for (int m = (k + 1); m < blk_m; m++) {

      Numeric *A_mk = &A[BLK_POS(m, k, blk_n)];
      for (int i = 0; i < BLK_SIZE; i++) {
        T[i] = 0;  
      }

      res = dtsqt2(&handle, A_kk, A_mk, T, Rbind);
      CHECK_ZERO_ERROR_RETURN(res, "Failed to compute dtsqt2");

      for (int n = (k + 1); n < blk_n; n++) {
        Numeric *A_kn = &A[BLK_POS(k, n, blk_n)];
        Numeric *A_mn = &A[BLK_POS(m, n, blk_n)];

        res = dssrfb(A_kn, A_mn, A_mk, T, Q, Q_);
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
  * @param handle cuBLAS handle
  * @param A m-by-n matrix to multiply and override. Adjacent to the source tile of Q.
  * @param Y m-by-n Hessianberg matrix holding householder vectors.
  * @param T n-by-n matrix.
  *
  * res = dlarfb(&handle, A_kn, A_kk, T, Q, Q_, BLK_LEN, BLK_LEN);
  */
__global__ void dlarfb_kernel(Numeric *M, int lbdm, int k, Numeric *T) {
    Numeric Q[BLK_SIZE];
    Numeric Q_[BLK_SIZE];

    Numeric *M_kk = &M[BLK_POS(k, k, lbdm)];
    Numeric *M_kn = &M[BLK_POS(k, k + 1 + threadIdx.x, lbdm)];

    dlarfb(M_kn, M_kk, T, Q, Q_);
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

    blk_dgeqt2(M_kk, T);
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

  dssrfb(A_kn, A_mn, V, T, X, Y);
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

    res = dtsqt2(&handle, A_kk, A_mk, T, Rbind);
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
    house(x, v, n);
}

extern "C"
int
TileQR_house(cublasHandle_t *handle, Vector *in, Vector *out) {    
    house_kernel<<<1, 1>>>(in->data_d, out->data_d, in->nr_elems);
    return 0;
}

__global__ void blk_dgeqt2_kernel_test(Numeric *A, Numeric *T) {
    blk_dgeqt2(A, T);
}

extern "C"
int
TileQR_blk_dgeqt2(Matrix *A, Matrix *T) {    
    blk_dgeqt2_kernel_test<<<1, 1>>>(A->data_d, T->data_d);
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
