#include "BlockOperations.h"
#include "BlockMatrix.h"
#include "Block.h"
#include "error.h"
#include "lapacke.h"

void
Block_col_sums(Block block,
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
Block_sub_vec(Block block,
              double *vec)
{
    for (int i = 0; i < BLK_SIZE; i++) {
        block[i] -= vec[i / BLK_LEN];
    }
}

int
Block_DGEQT2(Block A,
             Block T1)
{
    int res;
    Vector B;
    Block Y;
    
    res = Block_init(Y);
    CHECK_ZERO_RETURN(res);
    res = Vector_init_zero(&B, BLK_LEN);
    CHECK_ZERO_RETURN(res);

    int N = BLK_LEN;
    int LDA = N;
    double *TAU = B.data;
    double *WORK = NULL;
    int LWORK = -1;

    /* QR Decomposition */
    LAPACK_dgeqrf(&N, &N, A, &LDA, TAU, WORK, &LWORK, &res);
    CHECK_ZERO_RETURN(res);

    char DIRECT = 'F';
    char STOREV = 'C';
    /* Forms matrix T */
    LAPACK_dlarft(&DIRECT, &STOREV, &N, &N, A, &N, TAU, T1, &N);
    return 0;
}
