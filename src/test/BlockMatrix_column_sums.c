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
    CHECK_ZERO_RETURN(res); 

    res = Vector_init_zero(&columnSums, matrix.nr_cols);
    CHECK_ZERO_RETURN(res);
    
    res = BlockMatrix_column_sums(&matrix, &columnSums, scalar);
    CHECK_ZERO_RETURN(res);

    double expectedOutput = constant * nrRows * scalar;
    bool equals = true;

    int i = 0;
    while (i < nrCols && equals) {
        //printf("%lf %lf", columnSums.data[i], expectedOutput);
    	equals = DoubleCompare(columnSums.data[i++], expectedOutput);
    }

    BlockMatrix_free(&matrix);
    Vector_free(&columnSums);
    return equals ? 0 : 1;
}