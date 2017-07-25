#include "Perf.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    RunAll();

    MPI_Finalize();
    return 0;
}