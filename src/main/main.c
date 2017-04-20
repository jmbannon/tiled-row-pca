#include "../Vector.h"
#include "../BlockMatrix.h"
#include "../BlockMatrixOperations.h"
#include "../BlockMatrixVectorOperations.h"
#include "../DistBlockMatrix.h"
#include "../DistBlockMatrixOperations.h"
#include "../DoubleBlock.h"
#include "../Timer.h"
#include "../error.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char** argv) {
    int rows = 30000;
    int cols = 20000;

    printf("Calculating current PCA work on sized %dx%d sequential matrix.\n", rows, cols);

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    int res = 0;

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    DistBlockMatrix mat;
    BlockMatrix *local_mat;
    Vector local_col_means;
    Vector global_col_means;

    // Allocation:
    ////////////////////////////////////////////////////////////////////////////

    // Init matrix on both host and device
    res = DistBlockMatrix_init_zero(&mat, rows, cols, world_size, world_rank);
    CHECK_ZERO_RETURN(res);

    res = DistBlockMatrix_init_device(&mat);
    CHECK_ZERO_RETURN(res);

    local_mat = &mat.local;

    // Init local column means on both host and device
    res = Vector_init(&local_col_means, cols);
    CHECK_ZERO_RETURN(res);

    res = Vector_init_device(&local_col_means);
    CHECK_ZERO_RETURN(res);

    // Init global column means on host
    res = Vector_init(&global_col_means, cols);
    CHECK_ZERO_RETURN(res);

    res = Vector_init_device(&global_col_means);
    CHECK_ZERO_RETURN(res);

    res = DistBlockMatrix_seq(&mat, world_rank);
    CHECK_ZERO_RETURN(res);

    // Computation:
    ////////////////////////////////////////////////////////////////////////////

    res = BlockMatrix_copy_host_to_device(local_mat);
    CHECK_ZERO_RETURN(res);

    // Compute column means in device
    res = BlockMatrix_device_column_sums(local_mat, &local_col_means, 1.0 / local_mat->nr_rows);
    CHECK_ZERO_RETURN(res);

    // Copy column means from device to host
    res = Vector_copy_device_to_host(&local_col_means);
    CHECK_ZERO_RETURN(res);

    // ReduceALL local column means into global column means
    res = DistBlockMatrix_host_global_column_means(&mat, &local_col_means, &global_col_means);
    CHECK_ZERO_RETURN(res);

    // Center local data around origin
    res = BlockMatrixVector_device_sub(local_mat, &local_col_means);
    CHECK_ZERO_RETURN(res);

    // TODO: Free memory
    MPI_Finalize();
    printf("Sanity check, we didn't error out!\n");
    return 0;
}

