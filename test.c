#include "Vector.h"
#include "BlockMatrix.h"
#include "DistBlockMatrix.h"
#include "DistBlockMatrixOperations.h"
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
    DistBlockMatrix_init_zero(&mat, 2555565, 10, world_size, world_rank);

    Vector global_col_means;
    res = Vector_init_zero(&global_col_means, mat.global.nr_cols);
    CHECK_ZERO_RETURN(res);
    
    res = DistBlockMatrix_seq(&mat, world_rank);
    CHECK_ZERO_RETURN(res);
    
    res = DistBlockMatrix_column_means(&mat, &global_col_means);
    CHECK_ZERO_RETURN(res);

/*
    for (int i = 0; i < mat.nr_nodes; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == world_rank) {
            printf("proc %s, rank %d out of %d\n", processor_name, world_rank, world_size); 
            BlockMatrix_print_blocks(&mat.local);
        }
    }
*/

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        printf("column means:\n");
        Vector_print_blocks(&global_col_means);
    }

    DistBlockMatrix_free(&mat, world_rank);
    MPI_Finalize();
    
    return 0;
}

