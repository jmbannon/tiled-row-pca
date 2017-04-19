#include "../Vector.h"
#include "../BlockMatrix.h"
#include "../DistBlockMatrix.h"
#include "../DistBlockMatrixOperations.h"
#include "../DoubleBlock.h"
#include "../Timer.h"
#include "../error.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int rows = 8;
    int cols = 8;

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
    DistBlockMatrix_init_zero(&mat, rows, cols, world_size, world_rank);

    res = DistBlockMatrix_seq(&mat, world_rank);
    CHECK_ZERO_RETURN(res);

    res = DistBlockMatrix_copy_host_to_device(&mat);
    CHECK_ZERO_RETURN(res);

    res = DistBlockMatrix_normalize(&mat);
    CHECK_ZERO_RETURN(res);

    DistBlockMatrix_free(&mat, world_rank);

    MPI_Finalize();
    return 0;
}

