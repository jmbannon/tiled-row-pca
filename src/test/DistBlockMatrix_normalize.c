#include "Tests.h"

#include "../DistBlockMatrixOperations.h"
#include "../DistBlockMatrix.h"
#include "../BlockMatrix.h"
#include "../Block.h"
#include "../BlockMatrixOperations.h"
#include "../BlockMatrixVectorOperations.h"
#include "../DistBlockMatrix.h"
#include "../BlockMatrix.h"
#include "../Vector.h"
#include "../error.h"
#include "../Timer.h"
#include "../constants.h"
#include "DoubleCompare.h"

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdbool.h>

int Test_DistBlockMatrix_normalize()
{
    int res;
    int nrRows = 8;
    int nrCols = 8;

    int constant = 1.0;
    int scalar = 2.0;

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    /////////////////////////////////////////////////

    DistBlockMatrix mat;
    Vector col_means;

    printf("are we where1");

    res = DistBlockMatrix_init_zero(&mat, nrRows, nrCols, world_size, world_rank);
    CHECK_ZERO_RETURN(res);

    printf("are we where1.5");

    res = Vector_init(&col_means, mat.global.nr_cols);
    CHECK_ZERO_RETURN(res);

    printf("are we where2");

    res = DistBlockMatrix_seq(&mat, world_rank);
    CHECK_ZERO_RETURN(res);

    DistBlockMatrix_print_blocks(&mat, world_rank);

    res = DistBlockMatrix_normalize(&mat);
    CHECK_ZERO_RETURN(res);

    res = DistBlockMatrix_column_means(&mat, &col_means);
    CHECK_ZERO_RETURN(res);

    printf("are we where");

    bool equals = true;
    int i = 0;
    while (i < nrCols && equals) {
    	equals = DoubleCompare(col_means.data[i], 0.0);
    	++i;
    }

    DistBlockMatrix_free(&mat, world_rank);
    Vector_free(&col_means);

    return equals ? 0 : 1;
}