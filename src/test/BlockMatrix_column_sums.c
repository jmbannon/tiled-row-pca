#include "Tests.h"

#include "../DistBlockMatrixOperations.h"
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

int Test_BlockMatrix_column_sums()
{
    int res;
    int nrRows = 400;
    int nrCols = 400;

    int constant = 1.0;
    int scalar = 2.0;

    /////////////////////////////////////////////////

    Vector columnSums;
    BlockMatrix matrix;

    res = BlockMatrix_init_constant(&matrix, nrRows, nrCols, constant);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init constant"); 

    res = BlockMatrix_init_device(&matrix);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init device");

    res = BlockMatrix_copy_host_to_device(&matrix);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy from host to device");

    res = Vector_init_zero(&columnSums, matrix.nr_cols);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init zero vector");

    res = Vector_init_device(&columnSums);
    CHECK_ZERO_ERROR_RETURN(res, "failed to init device vector");
    
    res = BlockMatrix_device_column_sums(&matrix, &columnSums, scalar);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to calculate BlockMatrix device column sums");

    res = Vector_copy_device_to_host(&columnSums);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy device to host");

    res = Vector_free_device(&columnSums);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to free device vector");

    double expectedOutput = constant * nrRows * scalar;
    bool equals = true;

    int i = 0;
    while (i < nrCols && equals) {
    	equals = DoubleCompare(columnSums.data[i++], expectedOutput);
    }

    BlockMatrix_free(&matrix);
    Vector_free(&columnSums);
    return equals ? 0 : 1;
}