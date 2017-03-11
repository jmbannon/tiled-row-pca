#include "Vector.h"
#include "BlockMatrix.h"
#include "DistBlockMatrix.h"
#include "DistBlockMatrixOperations.h"
#include "DoubleBlock.h"
#include "Timer.h"
#include "error.h"
#include "test/BlockTest.h"
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

    double *block;
    res = Block_init_seq(&block);
    CHECK_ZERO_RETURN(res);

    // Block_print(block);
    res = Block_zero_tri(block, false, true);
    Block_print(block);
    printf("nope\n");
    double *dbl_blk;
    DoubleBlock_init_rbind(&dbl_blk, block, block);
    DoubleBlock_print(dbl_blk);
    
    DistBlockMatrix mat;
    DistBlockMatrix_init_zero(&mat, 131072*16, 256, world_size, world_rank);
    // Vector vec;
    // Vector_init_zero(&vec, mat.global.nr_cols);

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
   
    //test_DGEQT3();
    //test_Block_init_rbind();
    //test_Block_tri();
    //test_DGEQT2();    
    MPI_Finalize();
    return 0;
}

