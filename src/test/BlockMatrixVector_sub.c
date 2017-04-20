#include "../BlockMatrixVectorOperations.h"
#include "../BlockMatrix.h"
#include "../constants.h"
#include "../BlockMatrixOperations.h"
#include "../BlockOperations.h"
#include "../Vector.h"
#include "../error.h"
#include "DoubleCompare.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdbool.h>

int Test_BlockMatrixVector_sub()
{
    int res;
    int nrRows = 400;
    int nrCols = 400;

    int constant = 5.5;
    const double expectedOutput = 0.0;

    /////////////////////////////////////////////////

    Vector columnSums;
    BlockMatrix matrix;

    res = BlockMatrix_init_constant(&matrix, nrRows, nrCols, constant);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init constant"); 

    res = BlockMatrix_init_device(&matrix);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init device");

    res = BlockMatrix_copy_host_to_device(&matrix);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy matrix from host to device");

    res = Vector_init_constant(&columnSums, matrix.nr_cols, constant);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to init zero vector");

    res = Vector_init_device(&columnSums);
    CHECK_ZERO_ERROR_RETURN(res, "failed to init device vector");

    res = Vector_copy_host_to_device(&columnSums);
    CHECK_ZERO_ERROR_RETURN(res, "failed to copy vector from host to device");
    
    res = BlockMatrixVector_device_sub(&matrix, &columnSums);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to calculate matrix minus some vector");

    res = BlockMatrix_device_column_sums(&matrix, &columnSums, 1.0);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to calculate matrix column sums");

    res = Vector_copy_device_to_host(&columnSums);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to copy vector from device to host");

    res = BlockMatrix_free_device(&matrix);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to free device vector");

    res = Vector_free_device(&columnSums);
    CHECK_ZERO_ERROR_RETURN(res, "Failed to free device vector");

    bool equals = true;

    int i = 0;
    while (i < nrCols && equals) {
    	equals = DoubleCompare(columnSums.data[i++], expectedOutput);
    }

    BlockMatrix_free(&matrix);
    Vector_free(&columnSums);
    return equals ? 0 : 1;
}