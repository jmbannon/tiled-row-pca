#include <stdio.h>
#include "BlockOperations.h"
#include "BlockMatrix.h"
#include "Block.h"
#include "error.h"
#include "lapacke.h"

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
    res = Block_init_rbind(VRout, VRin, T1inout);
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

