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