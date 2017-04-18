#include "DistBlockMatrixOperations.h"
#include "BlockMatrix.h"
#include "Block.h"
#include "BlockMatrixOperations.h"
#include "BlockMatrixVectorOperations.h"
#include "DistBlockMatrix.h"
#include "Vector.h"
#include "error.h"
#include "Timer.h"
#include "constants.h"
#include <mpi.h>
#include <omp.h>
#include <stdio.h>

int
DistBlockMatrix_column_means(DistBlockMatrix *mat,
                             Vector *col_means)
{
    int res;
    Vector local_col_means;

    res = Vector_init_zero(&local_col_means, mat->global.nr_cols);
    CHECK_ZERO_RETURN(res);
    
    //res = BlockMatrix_column_sums(&mat->local, &local_col_means, 1.0 / mat->global.nr_rows);
    res = BlockMatrix_column_sums(&mat->local, &local_col_means, 1.0 / mat->global.nr_rows);
    CHECK_ZERO_RETURN(res);

    BlockMatrix_print_blocks(&mat->local);

    Vector_print_blocks(&local_col_means);

    MPI_Allreduce(local_col_means.data,
                  col_means->data,
                  col_means->nr_blk_elems * BLK_LEN,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    Vector_free(&local_col_means);
    return 0;
}

int
DistBlockMatrix_normalize(DistBlockMatrix *mat)
{
    int res;
    Vector col_means;
    res = Vector_init(&col_means, mat->global.nr_cols);
    CHECK_ZERO_RETURN(res);
    
    res = DistBlockMatrix_column_means(mat, &col_means);
    CHECK_ZERO_RETURN(res);
    
    Timer timer;
    Timer_start(&timer);    
    res = BlockMatrixVector_sub(&mat->local, &col_means);
    Timer_end(&timer);    
    printf("Col sums time: %lf\n", Timer_dur_sec(&timer));    
    
    Vector_free(&col_means);
    return res;
}
