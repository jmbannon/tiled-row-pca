#include "DistBlockMatrixOperations.h"
#include "BlockMatrix.h"
#include "BlockMatrixOperations.h"
#include "DistBlockMatrix.h"
#include "Vector.h"
#include "error.h"
#include <mpi.h>


int
DistBlockMatrix_column_means(DistBlockMatrix *mat,
                             Vector *col_means)
{
    int res;
    Vector local_col_means;

    res = Vector_init_zero(&local_col_means, mat->global.nr_cols);
    CHECK_ZERO_RETURN(res);
    
    res = BlockMatrix_column_sums(&mat->local, &local_col_means, 1.0 / mat->global.nr_rows);
    CHECK_ZERO_RETURN(res);
    
    MPI_Reduce(local_col_means.data,
               col_means->data,
               col_means->nr_blk_elems * BLK_LEN,
               MPI_DOUBLE,
               MPI_SUM,
               0, MPI_COMM_WORLD);

    Vector_free(&local_col_means);
    return 0;
}

