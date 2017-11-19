#include "../TileQR_Operations.h"
#include "../constants.h"
#include "../error.h"
#include "../Vector.h"
#include "../Matrix.h"
#include "../BlockMatrix.h"
#include <cuda.h>
#include <cuda_runtime.h>
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

  for (int i = 0; i < BLK_LEN; i++) {

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

  for (int i = 0; i < BLK_LEN; i++) {

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

// householder vec mult
__device__ void gevtm(int m, int p,
                      const Numeric alpha,
                      const Numeric *A, int lda,
                      const Numeric *B, int ldb,
                      Numeric *C, int ldc) {
  register int c_idx;
  for (int i = 0; i < m; i++) {

      c_idx = MAT_POS(i, 0, ldc);
      C[c_idx] = A[MAT_POS(0, i, lda)];

      for (int k = 1; k < p; k++) {
          C[c_idx] += A[MAT_POS(k, i, lda)] * B[MAT_POS(k, 0, ldb)];
      }
      C[c_idx] *= alpha;
  }
}

__device__ void gemtm(int m, int n, int p,
                      const Numeric alpha,
                      const Numeric *A, int lda,
                      const Numeric *B, int ldb,
                      const Numeric beta,
                      Numeric *C, int ldc) {
  register int c_idx;
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
  register int c_idx;
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
  register int c_idx;
  for (int i = 0; i < BLK_LEN; i++) {

    for (int j = 0; j < BLK_LEN; j++) {

      c_idx = MAT_POS(i, j, BLK_LEN);
      C[c_idx] = (beta == 0.0) ? 0.0 : C[c_idx] * beta;

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
  register int c_idx;
  for (int i = 0; i < BLK_LEN; i++) {

    for (int j = 0; j < BLK_LEN; j++) {

      c_idx = MAT_POS(i, j, ldc);
      C[c_idx] = (beta == 0.0) ? 0.0 : C[c_idx] * beta;

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
  register int c_idx;

  for (int i = 0; i < BLK_LEN; i++) {

    for (int j = 0; j < BLK_LEN; j++) {

      c_idx = MAT_POS(i, j, ldc);
      C[c_idx] = (beta == 0.0) ? 0.0 : C[c_idx] * beta;

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
__device__ void house(Numeric *x, Numeric *v, int n)
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
__device__ void house_row(Numeric *A, Numeric *v, Numeric *beta, Numeric *w, int m, int n, int ldm)
{
    norm(m, v, beta);

    *beta = -2.0 / (*beta * *beta);

    // Computes w
    gemtv(m, n, *beta, A, ldm, v, w);

    // Annihilate column of A: A = A + v * t(w)
    gemm(m, n, 1, 1.0, v, m, w, 1, 1.0, A, ldm);
}

/**
  * Produces an upper triangular matrix R, unit lower triangular matrix V that contains n Householder reflectors.
  * R and V are written on the memory area used for A.
  *
  * @param A upper triangular matrix R. Lower triangular contains partial householder reflectors:
  *        all diagonal elements should be 1 to represent full householder reflector.
  * @param w Work-space vector.
  * @param v Work-space vector.
  * @param store_house True if householder vectors should be stored in lower-triangular portion of output. False otherwise.
  */
__device__ void house_qr(Numeric *A, Numeric *beta, Numeric *w, Numeric *v, int m, int n)
{
  register int pos;
  for (int j = 0; j < n; j++) {
    pos = MAT_POS(j, j, m);

    house(&A[pos], &v[j], m - j);
    house_row(&A[pos], &v[j], &beta[j], &w[j], m - j, n - j, m);

    // Copies householder vector into lower triangular portion of A
    if (j < m) {
      for (int i = 1; i < m - j; i++) {
        A[pos + i] = v[j + i];
      }
    }
  }
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
__device__ void house_yt(Numeric *Y, Numeric *T, Numeric *beta, int m, int n)
{
  Numeric alpha;
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
    gevtm(j, m - j, alpha, &Y[y_idx], m, &Y[v_idx], m, &T[z_idx], n);

    // Computes T * z' using a triangular matrix-vector multiplication routine.
    trmv(j, T, n, &T[z_idx]);

    T[MAT_POS(j, j, n)] = beta[j];
  }
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
__device__ void dblk_dgeqt2(Numeric *A, Numeric *T)
{
  Numeric w[DBL_BLK_LEN];
  Numeric v[DBL_BLK_LEN];
  Numeric beta[BLK_LEN];

  house_qr(A, beta, w, v, DBL_BLK_LEN, BLK_LEN);
  house_yt(A, T, beta, DBL_BLK_LEN, BLK_LEN);
}

__device__ void blk_dgeqt2(Numeric *A, Numeric *T)
{
  Numeric w[BLK_LEN];
  Numeric v[BLK_LEN];
  Numeric beta[BLK_LEN];

  house_qr(A, beta, w, v, BLK_LEN, BLK_LEN);
  house_yt(A, T, beta, BLK_LEN, BLK_LEN);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// DTSQT2
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
  * Performs DGEQT2 on a row-binded matrix rbind(R, A).
  *
  * @param R n-by-n upper-triangular matrix to row-bind. May have non-zero elements in lower diagonal. Stores output R matrix from DGEQT2 here.
  * @param A n-by-n matrix to row-bind. Stores lower portion of householder vectors from DGEQT2 in here. The 'hessianberg' portion of the householder
  *          vectors is an identity matrix, so there is no need to store that.
  * @param T n-by-n output matrix.
  * @param RA_rowbind 2n-by-n work matrix to store the row-bind and compute DGEQT2 on.
  */
__device__ void dtsqt2(Numeric *R, Numeric *A, Numeric *T, Numeric *RA_rowbind)
{
  // Stores R into upper-portion of RA_rowbind. Zeroes lower-triangular portion.
  for (int j = 0; j < BLK_LEN; j++) {
    for (int i = 0; i < BLK_LEN; i++) {
      RA_rowbind[MAT_POS(i, j, DBL_BLK_LEN)] = (i > j) ? 0.0 : R[MAT_POS(i, j, BLK_LEN)];
    }
  }

  // Stores A into lower-portion of RA_rowbind.
  for (int j = 0; j < BLK_LEN; j++) {
    for (int i = 0; i < BLK_LEN; i++) {
      RA_rowbind[MAT_POS(BLK_LEN + i, j, DBL_BLK_LEN)] = A[MAT_POS(i, j, BLK_LEN)];
    }
  }

  dblk_dgeqt2(RA_rowbind, T);

  // Stores output R matrix into upper-triangular portion of R
  for (int j = 0; j < BLK_LEN; j++) {
    for (int i = 0; i <= j; i++) {
      R[MAT_POS(i, j, BLK_LEN)] = RA_rowbind[MAT_POS(i, j, DBL_BLK_LEN)];
    }
  }

  // Stores output householder vectors into A.
  for (int j = 0; j < BLK_LEN; j++) {
    for (int i = 0; i < BLK_LEN; i++) {
      A[MAT_POS(i, j, BLK_LEN)] = RA_rowbind[MAT_POS(BLK_LEN + i, j, DBL_BLK_LEN)];
    }
  }
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
__device__ void dssrfb(Numeric *A_kj,
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
  for (int i = 0; i < BLK_SIZE; i++) {
    A_kj[i] += X[i];
  }

  // A_ij = (V * Z) + A_ij
  blk_gemm(alpha, V, X, alpha, A_ij);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// DLAFRB
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
  * Computes Q = I + (Y * T * t(Y))
  *
  * @param Y m-by-n Hessianberg matrix containing householder vectors in lower portion.
  * @param T n-by-n Upper-triangular matrix
  * @param Q_ m-by-n work matrix to store Y * T
  * @return Q m-by-m output matrix
  */
__device__ void house_qr_q(Numeric *Y, Numeric *T, Numeric *Q, Numeric *Q_)
{    
    // Calculates Q' = Y * T
    blk_trmml(1.0, Y, T, Q_);

    // Calculates Q = Q' * t(Y)
    //              = Y * T * t(Y)
    blk_trmmr(1.0, Q_, Y, Q);

    // Calculates Q = I + Q
    //              = I + (Y * T * t(Y))
    for (int i = 0; i < BLK_SIZE; i += BLK_LEN + 1) {
      Q[i] += 1.0;
    }
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
__device__ void dlarfb(Numeric *A, Numeric *Y, Numeric *T, Numeric *Q, Numeric *Q_)
{
  house_qr_q(Y, T, Q, Q_);
  gemtm(1.0, Q, BLK_LEN, A, BLK_LEN, 0.0, Q_, BLK_LEN);

  #pragma unroll
  for (int i = 0; i < BLK_SIZE; i++) {
    A[i] = Q_[i];
  }
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

  res = cudaMalloc(&T, BLK_SIZE_MEM);
  CHECK_SUCCESS_RETURN(res);

  res = cudaMalloc(&Rbind, 2 * BLK_SIZE_MEM);
  CHECK_SUCCESS_RETURN(res);

  res = cudaMalloc(&Q, BLK_SIZE_MEM);
  CHECK_SUCCESS_RETURN(res);

  res = cudaMalloc(&Q_, BLK_SIZE_MEM);
  CHECK_SUCCESS_RETURN(res);

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

    blk_dgeqt2(A_kk, T);

    for (int n = (k + 1); n < blk_n; n++) {

      Numeric *A_kn = &A[BLK_POS(k, n, blk_n)];

      dlarfb(A_kn, A_kk, T, Q, Q_);
    }

    for (int m = (k + 1); m < blk_m; m++) {

      Numeric *A_mk = &A[BLK_POS(m, k, blk_n)];
      for (int i = 0; i < BLK_SIZE; i++) {
        T[i] = 0;  
      }

      dtsqt2(A_kk, A_mk, T, Rbind);

      for (int n = (k + 1); n < blk_n; n++) {
        Numeric *A_kn = &A[BLK_POS(k, n, blk_n)];
        Numeric *A_mn = &A[BLK_POS(m, n, blk_n)];

        dssrfb(A_kn, A_mn, A_mk, T, Q, Q_);

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

__global__ void dgeqt2_master(Numeric *M, int lbdm, int ki, int nr_blk_cols) {
  __shared__ Numeric T[BLK_SIZE];
  __shared__ Numeric A_mk[BLK_SIZE];
  // __shared__ Numeric M_kk[BLK_SIZE];

  Numeric A_kn[BLK_SIZE];
  Numeric A_mn[BLK_SIZE];
  Numeric Rbind[2 * BLK_SIZE];
  Numeric *X = Rbind;
  Numeric *Y = &Rbind[BLK_SIZE];

  int k = ki - blockIdx.x;
  int m = ki + blockIdx.x;

  for (int tr = threadIdx.x; tr < BLK_SIZE; tr += blockDim.x) {
    T[tr] = 0;
    // M_kk[tr] = M[BLK_POS(k, k, nr_blk_cols) + tr];
    A_mk[tr] = M[BLK_POS(m, k, nr_blk_cols) + tr];
  }

  __syncthreads();

  int nr_blks_to_process = nr_blk_cols - k - 1;

  if (blockIdx.x == 0) {
    // Numeric *A_kk = &M[BLK_POS(k, k, nr_blk_cols)];

    if (threadIdx.x == 0) {
        blk_dgeqt2(A_mk, T);
    }
    __syncthreads();

    if (nr_blks_to_process > 0) {

      for (int i = k + 1 + threadIdx.x; i < nr_blk_cols; i += blockDim.x) {

        for (int tr = 0; tr < BLK_SIZE; tr++) {
          A_kn[tr] = M[BLK_POS(k, i, nr_blk_cols) + tr];
        }

        dlarfb(A_kn, A_mk, T, X, Y);

        for (int tr = 0; tr < BLK_SIZE; tr++) {
          M[BLK_POS(k, i, nr_blk_cols) + tr] = A_kn[tr];
        }
      }
      __syncthreads();
    }

    for (int tr = threadIdx.x; tr < BLK_SIZE; tr += blockDim.x) {
      M[BLK_POS(m, k, nr_blk_cols) + tr] = A_mk[tr];
    }
  } else {

    if (threadIdx.x == 0) {
      // Numeric Rbind[2 * BLK_SIZE];
      Numeric *A_kk = &M[BLK_POS(k, k, nr_blk_cols)];
      dtsqt2(A_kk, A_mk, T, Rbind);
    }

    __syncthreads();

    if (nr_blks_to_process > 0) {
      for (int i = k + 1 + threadIdx.x; i < nr_blk_cols; i += blockDim.x) {

        for (int tr = 0; tr < BLK_SIZE; tr++) {
          A_kn[tr] = M[BLK_POS(k, i, nr_blk_cols) + tr];
          A_mn[tr] = M[BLK_POS(m, i, nr_blk_cols) + tr];
        }

        dssrfb(A_kn, A_mn, A_mk, T, X, Y);

        for (int tr = 0; tr < BLK_SIZE; tr++) {
          M[BLK_POS(k, i, nr_blk_cols) + tr] = A_kn[tr];
          M[BLK_POS(m, i, nr_blk_cols) + tr] = A_mn[tr];
        }
      }
      __syncthreads();
    }
    
    for (int tr = threadIdx.x; tr < BLK_SIZE; tr += blockDim.x) {
      M[BLK_POS(m, k, nr_blk_cols) + tr] = A_mk[tr];
    }
  }
}

__global__ void dtsqt2_master(Numeric *M, int lbdm, int ki, int mi, int nr_blk_cols) {
  __shared__ Numeric T[BLK_SIZE];
  __shared__ Numeric A_mk[BLK_SIZE];
  __shared__ Numeric A_kk[BLK_SIZE];

  Numeric A_kn[BLK_SIZE];
  Numeric A_mn[BLK_SIZE];
  Numeric Rbind[2 * BLK_SIZE];
  Numeric *X = Rbind;
  Numeric *Y = &Rbind[BLK_SIZE];

  int k = ki - blockIdx.x;
  int m = mi + blockIdx.x;

  for (int tr = threadIdx.x; tr < BLK_SIZE; tr += blockDim.x) {
    A_mk[tr] = M[BLK_POS(m, k, lbdm) + tr];
    A_kk[tr] = M[BLK_POS(k, k, lbdm) + tr];
  }
  __syncthreads();

  int nr_blks_to_process = nr_blk_cols - k - 1;

  // Numeric *A_mk = &M[BLK_POS(m, k, lbdm)];

  if (threadIdx.x == 0) {
    // Numeric Rbind[2 * BLK_SIZE];
    // Numeric *A_kk = &M[BLK_POS(k, k, lbdm)];
    dtsqt2(A_kk, A_mk, T, Rbind);
  }
  __syncthreads();

  if (nr_blks_to_process > 0) {
      for (int i = k + 1 + threadIdx.x; i < nr_blk_cols; i += blockDim.x) {
        // printf("dssrfb %d: k=%d %d %d\n", threadIdx.x, k, m, k + start_col + i);

        for (int tr = 0; tr < BLK_SIZE; tr++) {
          A_kn[tr] = M[BLK_POS(k, i, nr_blk_cols) + tr];
          A_mn[tr] = M[BLK_POS(m, i, nr_blk_cols) + tr];
        }
        
        dssrfb(A_kn, A_mn, A_mk, T, X, Y);

        for (int tr = 0; tr < BLK_SIZE; tr++) {
          M[BLK_POS(k, i, nr_blk_cols) + tr] = A_kn[tr];
          M[BLK_POS(m, i, nr_blk_cols) + tr] = A_mn[tr];
        }
      }
      
  }

  __syncthreads();

  for (int tr = threadIdx.x; tr < BLK_SIZE; tr += blockDim.x) {
    M[BLK_POS(m, k, lbdm) + tr] = A_mk[tr];
    M[BLK_POS(k, k, lbdm) + tr] = A_kk[tr];
  }
}

int powdown(int x) {
  const int max = 256;
  int ans = 1;
  while ((ans * 2) <= x && (ans * 2) <= max) ans *= 2;

  return ans;
}

extern "C"
int
BlockMatrix_TileQR_multi_thread(BlockMatrix *BlkM) {
  int blk_m = BlkM->nr_blk_rows;
  int blk_n = BlkM->nr_blk_cols;

  int min_blk_d = blk_m > blk_n ? blk_n : blk_m;
  Numeric *M = BlkM->data_d;

  int blocks = 1;
  int threads = powdown(blk_n - 1);
  int shared_mem = threads * BLK_SIZE_MEM * 2;

  // printf("blocks: %d, threads: %d\n", 1, blk_n - 1);
  dgeqt2_master<<<blocks, threads>>>(M, blk_n, 0, blk_n);

  // printf("blocks: %d, threads: %d\n", 1, blk_n - 1);
  dtsqt2_master<<<blocks, threads>>>(M, blk_n, 0, 1, blk_n);

  int i = 1;
  while (i < min_blk_d && i < blk_n) {
    blocks = (i + i) < blk_m ? i + 1 : blk_m - i;
    threads = powdown(blk_n - i - 1 + blocks);
    shared_mem = threads * BLK_SIZE_MEM * 2;

    // printf("blocks: %d, threads: %d\n", blocks, blk_n - i - 1 + blocks);
    dgeqt2_master<<<blocks, threads>>>(M, blk_n, i, blk_n);

    blocks = (i + i + 1) < blk_m ? i + 1 : blk_m - i - 1;
    threads = powdown(blk_n - i - 1 + blocks);
    shared_mem = threads * BLK_SIZE_MEM * 2;

    // printf("blocks: %d, threads: %d\n", blocks, blk_n - i - 1 + blocks);
    dtsqt2_master<<<blocks, threads>>>(M, blk_n, i, i + 1, blk_n);
    ++i;
  }

  ++i;
  while (i < blk_m) {
    blocks = (i + blk_n) <= blk_m ? min_blk_d : blk_m - i;
    threads = powdown(blocks);
    shared_mem = threads * BLK_SIZE_MEM * 2;

    // printf("blocks: %d, threads: %d\n", blocks, blocks);
    dtsqt2_master<<<blocks, threads>>>(M, blk_n, blk_n - 1, i, blk_n);
    ++i;
  }

  cudaDeviceSynchronize();

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
TileQR_house(Vector *in, Vector *out) {    
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
