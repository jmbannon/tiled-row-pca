#include "Vector.h"
#include "BlockMatrix.h"
#include "DistBlockMatrix.h"
#include "DistBlockMatrixOperations.h"
#include "Timer.h"
#include "error.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    int res;

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    
    DistBlockMatrix mat;
    DistBlockMatrix_init_zero(&mat, 131072*world_size, 256, world_size, world_rank);
    Vector vec;
    Vector_init_zero(&vec, mat.global.nr_cols);

    res = DistBlockMatrix_seq(&mat, world_rank);
    CHECK_ZERO_RETURN(res);

    Timer timer;
    MPI_Barrier(MPI_COMM_WORLD);
    Timer_start(&timer);    
    res = DistBlockMatrix_normalize(&mat);
    MPI_Barrier(MPI_COMM_WORLD);
    Timer_end(&timer);
    CHECK_ZERO_RETURN(res);

    if (world_rank == 0) {
        printf("Matrix normalization: %lf\n", Timer_dur_sec(&timer));
    }
  
    //DistBlockMatrix_print_blocks(&mat, world_rank);  
    DistBlockMatrix_free(&mat, world_rank);
    
    test_1();    
    MPI_Finalize();
    return 0;
}

