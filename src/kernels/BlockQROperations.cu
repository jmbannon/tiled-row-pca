#include "../BlockQROperations.h"
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
    cublasHandle_t handle2;
    int res = cublasCreate(&handle2);
    CHECK_CUBLAS_RETURN(res, "Failed to create handle");

    Numeric x_norm;
    res = 0;

    // Copies x into v and calculates the norm
    #if FLOAT_NUMERIC
    	res = cublasScopy(handle2, n, x, 1, v, 1);
      CHECK_CUBLAS_RETURN(res, "Failed to copy vector x");

    	res = cublasSnrm2(handle2, n, x, 1, &x_norm);
    #else
    	res = cublasDcopy(handle2, n, x, 1, v, 1);
      CHECK_CUBLAS_RETURN(res, "Failed to copy vector x");

    	res = cublasDnrm2(handle2, n, x, 1, &x_norm);
    #endif
    CHECK_CUBLAS_RETURN(res, "Failed to calculate norm of x");

    if (x_norm != 0) {
    	const Numeric sign = x[0] >= 0 ? 1.0 : -1.0;
    	const Numeric beta = 1.0 / (x[0] + (sign * x_norm));
    	#if FLOAT_NUMERIC
    		res = cublasSscal(handle2, n - 1, &beta, &v[1], 1);
    	#else
    		res = cublasDscal(handle2, n - 1, &beta, &v[1], 1);
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
    cublasHandle_t handle2;
    int res = cublasCreate(&handle2);

    res = 0;

    // Computes beta
    #if FLOAT_NUMERIC
      res = cublasSnrm2(handle2, m, v, 1, beta);
    #else
      res = cublasDnrm2(handle2, m, v, 1, beta);
    #endif
    CHECK_CUBLAS_RETURN(res, "Failed to compute beta");

    *beta = -2.0 / (*beta * *beta);

    // Computes w
    Numeric w_scalar = 0.0;
    #if FLOAT_NUMERIC
      res = cublasSgemv(handle2, CUBLAS_OP_T, m, n, beta, A, ldm, v, 1, &w_scalar, w, 1);
    #else
      res = cublasDgemv(handle2, CUBLAS_OP_T, m, n, beta, A, ldm, v, 1, &w_scalar, w, 1);
    #endif
    CHECK_CUBLAS_RETURN(res, "Failed to compute w");

    // Annihilate column of A: A = A + v * t(w)
    Numeric scalar = 1.0;
    #if FLOAT_NUMERIC
      res = cublasSgemm(handle2, CUBLAS_OP_N, CUBLAS_OP_N, m, n, 1, &scalar, v, m, w, 1, &scalar, A, ldm);
    #else
      res = cublasDgemm(handle2, CUBLAS_OP_N, CUBLAS_OP_N, m, n, 1, &scalar, v, m, w, 1, &scalar, A, ldm);
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
  cublasHandle_t handle2;
  int res = cublasCreate(&handle2);

  Numeric *v;

  res = cudaMalloc(&v, m * sizeof(Numeric));
  CHECK_SUCCESS_RETURN(res);

  for (int j = 0; j < n; j++) {
    int pos = MAT_POS(j, j, m);
    res = house(&handle2, &A[pos], &v[j], m - j);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to compute house");

    res = house_row(&handle2, &A[pos], &v[j], &beta[j], &w[j], m - j, n - j, m);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to compute house_row");

    printf("\nhouseholder vector for j=%d\n", j);
    for (int i = j; i < m; i++) {
      printf("%f ", v[i]);
    }
    printf("\ncurrent matrix:\n");
    for (int a = 0; a < m; a++) {
      for (int b = 0; b < n; b++) {
        printf("%f ", A[MAT_POS(a, b, m)]);
      }
      printf("\n");
    }
    printf("\n");

    // Copies householder vector into lower triangular portion of A
    if (store_house && j < m) {
      #if FLOAT_NUMERIC
        res = cublasScopy(handle2, m - j - 1, &v[j + 1], 1, &A[pos + 1], 1);
      #else
        res = cublasDcopy(handle2, m - j - 1, &v[j + 1], 1, &A[pos + 1], 1);
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
  * @see{https://www.cs.cornell.edu/cv/ResearchPDF/A%20Storage-Efficient%20WY%20Representation%20for%20Products%20of%20Householder%20Transformations.pdf}
  *
  * @param Y m-by-n matrix where lower-triangular + diag portion holds householder vectors and the upper-triangular portion holds the R matrix from QR.
  * @param T n-by-n output matrix to store T
  */
__device__ int house_yt(cublasHandle_t *handle, Numeric *Y, Numeric *T, Numeric *beta, int m, int n)
{
  cublasHandle_t handle2;
  int res = cublasCreate(&handle2);

  Numeric alpha;
  Numeric zero = 0.0;
  int v_idx;
  int z_idx;
  int y_idx;

  printf("beta values\n");
  for (int i = 0; i < n; i++) {
    printf("%f ", beta[i]);
  }
  printf("\n");

  printf("y_%d = \n", 0);
    for (int i = 0; i < m; i++) {
      printf("%f ", Y[i]);
    }
    printf("\n");


  T[0] = beta[0];
  for (int j = 1; j < n; j++) {
    alpha = beta[j];

    y_idx = MAT_POS(j, 0, m);
    v_idx = MAT_POS(j, j, m);
    z_idx = MAT_POS(0, j, n);

    printf("y_%d = \n", j);
    for (int i = 0; i < (m - j); i++) {
      printf("%f ", Y[v_idx + i]);
    }
    printf("\n");

    // Computes -2 * t(Y) * v_j = z' in an optimized way to ignore 0 elements in v_j. Stores it in z-location of T matrix.
    #if FLOAT_NUMERIC
      res = cublasSgemm(handle2, CUBLAS_OP_T, CUBLAS_OP_N, j, 1, m - j, &alpha, &Y[y_idx], m, &Y[v_idx], m, &zero, &T[z_idx], n);
    #else
      res = cublasDgemm(handle2, CUBLAS_OP_T, CUBLAS_OP_N, j, 1, m - j, &alpha, &Y[y_idx], m, &Y[v_idx], m, &zero, &T[z_idx], n);
    #endif

    printf("PRE we are at j=%d\n", j);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        printf("%f ", T[MAT_POS(i, j, n)]);
      }
      printf("\n");
    }

    // Computes T * z' using a triangular matrix-vector multiplication routine.
    #if FLOAT_NUMERIC
      res = cublasStrmv(handle2, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, j, T, n, &T[z_idx], 1);
    #else
      res = cublasDtrmv(handle2, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, j, T, n, &T[z_idx], 1);
    #endif

    printf("post we are at j=%d\n", j);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        printf("%f ", T[MAT_POS(i, j, n)]);
      }
      printf("\n");
    }

    T[MAT_POS(j, j, n)] = beta[j];
  }
  return 0;
}

__device__ int dgeqt2(cublasHandle_t *handle, Numeric *A, Numeric *T, int m, int n)
{
  cublasHandle_t handle2;
  int res = cublasCreate(&handle2);

  // Temporary work matrix
  Numeric *w;

  Numeric *beta;

  res = cudaMalloc(&w, m * sizeof(Numeric));
  CHECK_SUCCESS_RETURN(res);

  res = cudaMalloc(&beta, n * sizeof(Numeric));
  CHECK_SUCCESS_RETURN(res);

  res = house_qr(&handle2, A, beta, w, true, m, n);

  // Restore householder vectors for YT Generation. Store diag in work vector.
  int diag_idx;
  for (int i = 0; i < n; i++) {
    diag_idx = MAT_POS(i, i, m);
    w[i] = A[diag_idx];
    A[diag_idx] = 1.0;

  }

  res = house_yt(&handle2, A, T, beta, m, n);

  for (int i = 0; i < n; i++) {
    diag_idx = MAT_POS(i, i, m);
    A[diag_idx] = w[i];
  }

  return 0;
}

// Hessianberg matrix-matrix multiply
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
  #if FLOAT_NUMERIC
    res = cublasStrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, diag, k, n, &alpha, A, lda, B, ldb, C, ldc);
  #else
    res = cublasDtrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, diag, k, n, &alpha, A, lda, B, ldb, C, ldc);
  #endif 
  CHECK_CUBLAS_RETURN(res, "Triangle portion of Hessianberg matrix multiply failed");

  if (m != k) {
    printf("FORBIDDEN ZONE\n");
    #if FLOAT_NUMERIC
      res = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m - k, n, k, &alpha, &A[k], lda, B, ldb, &zero, &C[k], ldc);
    #else
      res = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m - k, n, k, &alpha, &A[k], lda, B, ldb, &zero, &C[k], ldc);
    #endif
    CHECK_CUBLAS_RETURN(res, "Rectangle portion of Hessianberg matrix multiply failed");
  }
  return 0;
}


//  n, m, n
// T (n x n) = A
// Y (m x n) = B

// Matrix-hessianberg matrix transpose multiply
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
  #if FLOAT_NUMERIC
    res = cublasStrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, diag, m, k, &alpha, B, ldb, A, lda, C, ldc);
  #else
    res = cublasDtrmm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, diag, m, k, &alpha, B, ldb, A, lda, C, ldc);
  #endif 
  CHECK_CUBLAS_RETURN(res, "Triangle portion of Hessianberg matrix multiply failed");

  if (n != k) {
    printf("FORBIDDEN ZONE\n");
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

__global__ void Block_house_kernel(cublasHandle_t *handle, Numeric *x, Numeric *v, int n) {
    house(handle, x, v, n);
}

extern "C"
int
Block_house(cublasHandle_t *handle, Vector *in, Vector *out) {    
    Block_house_kernel<<<1, 1>>>(handle, in->data_d, out->data_d, in->nr_elems);
    return 0;
}




__global__ void Block_house_qr_kernel(cublasHandle_t *handle, Numeric *A, int m, int n) {
    house_qr(handle, A, NULL, NULL, true, m, n);
}

extern "C"
int
Block_house_qr(cublasHandle_t *handle, Matrix *A) {    
    Block_house_qr_kernel<<<1, 1>>>(handle, A->data_d, A->nr_rows, A->nr_cols);
    return 0;
}


__global__ void Block_dgeqt2_kernel(cublasHandle_t *handle, Numeric *A, Numeric *T, int m, int n) {
    dgeqt2(handle, A, T, m, n);
}

extern "C"
int
Block_dgeqt2(cublasHandle_t *handle, Matrix *A, Matrix *T) {    
    Block_dgeqt2_kernel<<<1, 1>>>(handle, A->data_d, T->data_d, A->nr_rows, A->nr_cols);
    return 0;
}




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