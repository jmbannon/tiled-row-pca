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

    res = DistBlockMatrix_init_zero(&mat, nrRows, nrCols, world_size, world_rank);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init zero dist matrix");

    res = Vector_init(&col_means, mat.global.nr_cols);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init vector");

    res = DistBlockMatrix_seq(&mat, world_rank);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to sequentialize dist matrix");

    res = DistBlockMatrix_init_device(&mat);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init dist matrix device");

    res = DistBlockMatrix_copy_host_to_device(&mat);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy host to device dist matrix");

    res = DistBlockMatrix_normalize(&mat);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to normalize dist matrix");

    res = DistBlockMatrix_copy_host_to_device(&mat);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy host to device dist matrix after normalization");

    res = DistBlockMatrix_device_column_means(&mat, &col_means);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to calculate column means of dist matrix");

    bool equals = true;
    int i = 0;
    while (i < nrCols && equals) {
    	equals = DoubleCompare(col_means.data[i++], 0.0);
    }

    DistBlockMatrix_free(&mat, world_rank);
    Vector_free(&col_means);

    return equals ? 0 : 1;
}