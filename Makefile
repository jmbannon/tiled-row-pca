include make.inc

EXEC         = row-tile-pca
CC           = gcc-6
FCC          = gfortran
INCL         = -I./include
OMP_FLAG     = -fopenmp
OMPI_INCL    = -I${OMPI_DIR} -lmpi
LAPACK_INCL  = ${LAPACK_ARGS}
BLAS_INCL    = -L${BLAS_LIB_DIR} -llibblas.a
LAPACK_INCL2 = ${LAPACK_LIB_DIR}/liblapack.a -l${FCC}
LAPACK_INCL3 = ${LAPACK_CWRAPPER}/liblapack_cwrapper.a -l${FCC}
CUBLAS_INCL  = -L${CUBLAS_DIR} -lcublas
SRC          = $(wildcard src/*.c src/test/*.c)

FLAGS        = ${INCL} ${LAPACK_INCL2} ${LAPACK_INCL3} ${LAPACK_INCL} ${OMP_FLAG} ${OMPI_INCL} ${CUBLAS_INCL}

${EXEC}: ${SRC}
		${CC} -o $@ $^ ${FLAGS} ${DEFINES}

