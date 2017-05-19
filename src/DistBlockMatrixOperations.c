#include "BlockMatrix.h"
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
DistBlockMatrix_host_global_column_means(DistBlockMatrix *mat,
                                         Vector *local_col_means,
                                         Vector *global_col_means)
{
    // TODO: Have one that supports floats and doubles
    MPI_Allreduce(local_col_means->data,
                  global_col_means->data,
                  global_col_means->nr_blk_elems * BLK_LEN,
                  MPI_DOUBLE,
                  MPI_SUM,
                  MPI_COMM_WORLD);
    return 0;
}

int
DistBlockMatrix_device_column_means(DistBlockMatrix *mat,
                                    Vector *col_means)
{
    int res;
    Vector local_col_means;

    res = Vector_init(&local_col_means, mat->global.nr_cols);
    CHECK_ZERO_RETURN(res);

    res = Vector_init_device(&local_col_means);
    CHECK_ZERO_RETURN(res);
    
    res = BlockMatrix_device_column_sums(&mat->local, &local_col_means, 1.0 / mat->global.nr_rows);
    CHECK_ZERO_RETURN(res);

    res = Vector_copy_device_to_host(&local_col_means);
    CHECK_ZERO_RETURN(res);

    res = Vector_free_device(&local_col_means);
    CHECK_ZERO_RETURN(res);

    res = DistBlockMatrix_host_global_column_means(mat, &local_col_means, col_means);
    CHECK_ZERO_RETURN(res);

    Vector_free(&local_col_means);
    return 0;
}

int
DistBlockMatrix_global_normalize(DistBlockMatrix *mat)
{
    int res;
    Vector col_means;
    res = Vector_init(&col_means, mat->global.nr_cols);
    CHECK_ZERO_RETURN(res);
    
    res = DistBlockMatrix_device_column_means(mat, &col_means);
    CHECK_ZERO_RETURN(res);
    
    res = BlockMatrixVector_sub(&mat->local, &col_means);
    CHECK_ZERO_RETURN(res);
    
    Vector_free(&col_means);
    return res;
}
