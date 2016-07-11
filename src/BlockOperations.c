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

int
Block_DGEQT2(double *A,
             double *T1)
{
    int res = 0;
    int N = BLK_LEN;
    int LDA = N;

    /* QR Decomposition */
    LAPACK_dgeqrt2(&N, &N, A, &LDA, T1, &N,&res);
    CHECK_ZERO_RETURN(res);

    return 0;
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

int
Block_DTSQT2();
