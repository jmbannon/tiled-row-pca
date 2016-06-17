#include "PerfTests.h"
#include "DistBlockMatrix.h"
#include "BlockMatrix.h"
#include "DistBlockMatrixOperations.h"
#include <mpi.h>

/**
 * Initializes a sequential matrix and wraps a timer around
 * the Column Means function. Prints time on root node.
 *
 * @param rows Number of rows.
 * @param cols Number of cols.
 */
int
PerfTest_run_colmeans(int rows,
                      int cols)
{
    if (samples <= 0 || attrs <= 0) {
        return INVALID_DIMS; 
    }
    int world_size;
    int world_rank;
    
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    DistBlockMatrix mat;
    DistBlockMatrix_init_zero(&mat, rows, cols, world_size, world_rank);

    Vector global_col_means;
    res = Vector_init_zero(&global_col_means, mat.global.nr_cols);
    CHECK_ZERO_RETURN(res);
    
    res = DistBlockMatrix_seq(&mat, world_rank);
    CHECK_ZERO_RETURN(res);

    Timer timer;
    Timer_start(&timer);    
    res = DistBlockMatrix_column_means(&mat, &global_col_means);
    Timer_end(&timer);
    CHECK_ZERO_RETURN(res);

    if (world_rank == 0) {
        printf("Column means seconds: %lf\n", Timer_dur_sec(&timer));
    }
    return 0;
}

/**
 * USAGE:
 * ./<exec> <rows> <attributes>
 *
 * Executes column means on a sequential matrix of the 
 * specified dimensions to time its performance.
 */
int main(int argc, char **argv)
{
    if (argc == 4) {
    
    }
}
