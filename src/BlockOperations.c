#include <stdio.h>
#include <cblas.h>
#include "BlockOperations.h"
#include "BlockMatrix.h"
#include "DoubleBlock.h"
#include "Block.h"
#include "error.h"
#include "lapacke.h"
#include "constants.h"

void
Block_col_sums(double *block,
               double *col_sum)
{
    for (int i = 0; i < BLK_SIZE; i++) {
        col_sum[i / BLK_LEN] += block[i]; 
    }
}

/**
 * Subtracts block vector from block.
 */
void
Block_sub_vec(double *block,
              double *vec)
{
    for (int i = 0; i < BLK_SIZE; i++) {
        block[i] -= vec[i / BLK_LEN];
    }
}

static int
_Block_DGEQT2_internal(double *A,
                       double *T1,
                       int rows,
                       int cols)
{
    int res = 0;
    int M = rows;
    int N = cols;
    int LDA = rows > cols ? rows : cols;

    /* QR Decomposition */
    LAPACK_dgeqrt2(&M, &N, A, &LDA, T1, &N,&res);
    CHECK_ZERO_RETURN(res);

    return 0;
}

int
Block_DGEQT2(double *A,
             double *T1)
{
    return _Block_DGEQT2_internal(A, T1, BLK_LEN, BLK_LEN);
}

int
Block_DTSQT2(double *VRin,
             double *T1inout,
             double **VRout)
{
    int res = Block_zero_tri(VRin, false, false);
    CHECK_ZERO_RETURN(res);
    res = DoubleBlock_init_rbind(VRout, VRin, T1inout);
    CHECK_ZERO_RETURN(res);
    return _Block_DGEQT2_internal(*VRout, T1inout, BLK_LEN * 2, BLK_LEN);
}

int
Block_DLARFB(double *A,
             double *VR,
             double *T1)
{
    char SIDE = 'L';
    char TRANS = 'T';
    char DIRECT = 'F';
    char STOREV = 'C';
    int N = BLK_LEN;
    double *V = VR;
    double *T = T1;
    double *C = A;
    double *WORK = malloc(BLK_SIZE * sizeof(double));
    int WLDA = BLK_SIZE;
    CHECK_MALLOC_RETURN(WORK);
    
    LAPACK_dlarfb(&SIDE, &TRANS, &DIRECT, &STOREV, &N, &N, &N, V, &N, T, &N, C, &N, WORK, &N);

    free(WORK);
    return 0;
}


/** 
 * Assumes column-major
 * C := alpha * op( A ) * op( B ) + beta*C
 *
 */
static inline void
_cblas_dgemm(double *A,
             double *B,
             double *C,
             int A_nr_rows,
             int A_nr_cols,
             int B_nr_cols,
             int A_op,
             int B_op,
             int A_ld,
             int B_ld,
             int C_ld,
             double alpha,
             double beta)
{
    cblas_dgemm(CblasColMajor, A_op, B_op, A_nr_rows, B_nr_cols, A_nr_cols, alpha, A, A_ld, B, B_ld, beta, C, C_ld);
}


int
Block_DSSRFB3(double *A_kn,
              double *A_mn,
              double *V_mk,
              double *T1_mk)
{
    int res;
    double *tmp_diag_blk;
    double *V_mk_orig;
    double *dbl_diag;
    double *V_mk_T1_tmp;
    double *A_kn_A_mn;

    // TODO: Create a function that inits a diag on top of an existing block
    res = Block_init(&tmp_diag_blk);
    CHECK_ZERO_RETURN(res);
    res = DoubleBlock_init_rbind(&V_mk_orig, tmp_diag_blk, V_mk);
    CHECK_ZERO_RETURN(res);
    // ODOT

    res = Block_init(&V_mk_T1_tmp);
    CHECK_ZERO_RETURN(res);

    res = DoubleBlock_init_diag(&dbl_diag);
    CHECK_ZERO_RETURN(res);

    res = DoubleBlock_init_rbind(&A_kn_A_mn, A_kn, A_mn);
    CHECK_ZERO_RETURN(res);

    // V_mk_T1_tmp := V_mk_orig * T1_mk
    _cblas_dgemm(V_mk_orig, T1_mk, V_mk_T1_tmp, DBL_BLK_LEN, BLK_LEN, BLK_LEN, CBLAS_NO_TRANS, CBLAS_NO_TRANS, DBL_BLK_LEN, BLK_LEN, BLK_LEN, 1.0, 0.0);

    // dbl_diag := V_mk_T1_tmp * t(V_mk_orig) + dbl_diag
    _cblas_dgemm(V_mk_T1_tmp, V_mk_orig, dbl_diag, DBL_BLK_LEN, BLK_LEN, BLK_LEN, CBLAS_NO_TRANS, CBLAS_TRANS, DBL_BLK_LEN, BLK_LEN, DBL_BLK_LEN, 1.0, 1.0);

    // dbl_diag := t(dbl_diag) * A_kn_A_mn + (0 * dbl_diag)
    _cblas_dgemm(dbl_diag, A_kn_A_mn, dbl_diag, DBL_BLK_LEN, DBL_BLK_LEN, BLK_LEN, CBLAS_TRANS, CBLAS_TRANS, DBL_BLK_LEN, DBL_BLK_LEN, DBL_BLK_LEN, 1.0, 0.0);

    // free memory, split A_kn_A_mn
}


