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
    int res;
    Numeric x_norm;

    // Copies x into v and calculates the norm
    #if FLOAT_NUMERIC
    	res = cublasScopy(*handle, n, x, 1, v, 1);
      CHECK_CUBLAS_RETURN(res, "Failed to copy vector x");

    	res = cublasSnrm2(*handle, n, x, 1, &x_norm);
    #else
    	res = cublasDcopy(*handle, n, x, 1, v, 1);
      CHECK_CUBLAS_RETURN(res, "Failed to copy vector x");

    	res = cublasDnrm2(*handle, n, x, 1, &x_norm);
    #endif
    CHECK_CUBLAS_RETURN(res, "Failed to calculate norm of x");

    if (x_norm != 0) {
    	const Numeric sign = x[0] >= 0 ? 1.0 : -1.0;
    	const Numeric beta = 1.0 / (x[0] + (sign * x_norm));
    	#if FLOAT_NUMERIC
    		res = cublasSscal(*handle, n - 1, &beta, &v[1], 1);
    	#else
    		res = cublasDscal(*handle, n - 1, &beta, &v[1], 1);
    	#endif
      CHECK_CUBLAS_RETURN(res, "Failed scale vector v");
    }
    v[0] = 1.0;

    return res;
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
    int res;

    // Computes beta
    #if FLOAT_NUMERIC
      res = cublasSnrm2(*handle, m, v, 1, beta);
    #else
      res = cublasDnrm2(*handle, m, v, 1, beta);
    #endif
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
      #if FLOAT_NUMERIC
        res = cublasScopy(*handle, m - j - 1, &v[j + 1], 1, &A[pos + 1], 1);
      #else
        res = cublasDcopy(*handle, m - j - 1, &v[j + 1], 1, &A[pos + 1], 1);
      #endif
      CHECK_CUBLAS_RETURN(res, "Failed to copy householder vector into lower-triangular portion of A");
    }
  }
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

  return 0;
}

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
      RA_rowbind[MAT_POS(2*i, j, RArows)] = A[MAT_POS(i, j, n)];
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
      A[MAT_POS(i, j, n)] = RA_rowbind[MAT_POS(2*i, j, RArows)];
    }
  }

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



__global__ void dgeqt2_kernel(Numeric *A, Numeric *T, int m, int n) {
    cublasHandle_t handle;
    int res = cublasCreate(&handle);

    dgeqt2(&handle, A, T, m, n);
}

extern "C"
int
TileQR_dgeqt2(cublasHandle_t *handle, Matrix *A, Matrix *T) {    
    dgeqt2_kernel<<<1, 1>>>(A->data_d, T->data_d, A->nr_rows, A->nr_cols);
    return 0;
}




