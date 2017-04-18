#include "../Vector.h"
#include "../BlockMatrix.h"
#include "../DistBlockMatrix.h"
#include "../DistBlockMatrixOperations.h"
#include "../DoubleBlock.h"
#include "../Timer.h"
#include "../error.h"
#include "Tests.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    TestAll();

    MPI_Finalize();
    return 0;
}