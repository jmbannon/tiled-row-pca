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
    DistBlockMatrix_init_zero(&mat, 262144*world_size, 128, world_size, world_rank);

    res = DistBlockMatrix_seq(&mat, world_rank);
    CHECK_ZERO_RETURN(res);

    Timer timer;
    Timer_start(&timer);    
    res = DistBlockMatrix_normalize(&mat);
    Timer_end(&timer);
    CHECK_ZERO_RETURN(res);

    if (world_rank == 0) {
        printf("Column means seconds: %lf\n", Timer_dur_sec(&timer));
    }
  
    //DistBlockMatrix_print_blocks(&mat, world_rank);  
    DistBlockMatrix_free(&mat, world_rank);
    
    test_1();    
    MPI_Finalize();
    return 0;
}

