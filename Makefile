# == Edit to your configuration =======

WORKING_DIR  = /Users/jb/workspace/pca2
OMPI_DIR     = /usr/local/Cellar/open-mpi/1.10.2_1/include
BLAS_LIB_DIR = /usr/local/lib
LAPACK_ARGS  = -framework Accelerate

# ======================================

EXEC         = row-tile-pca
CC           = gcc-6
INCL         = -I./include
OMP_FLAG     = -fopenmp
OMPI_INCL    = -I${OMPI_DIR} -lmpi
LAPACK_INCL  = ${LAPACK_ARGS}
BLAS_INCL    = -L${BLAS_LIB_DIR} -llibblas.a
SRC          = $(wildcard src/*.c)

FLAGS        = ${INCL} ${LAPACK_INCL} ${OMP_FLAG} ${OMPI_INCL}

${EXEC}: ${SRC}
		${CC} -o $@ $^ ${FLAGS} ${DEFINES}

