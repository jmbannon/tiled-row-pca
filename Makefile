include make.inc

# - CC ----------------------------------------------------------------

EXEC         = row-tile-pca
CC           = nvcc -m64
FCC          = gfortran

# - Flags -------------------------------------------------------------

LOCAL        = -I./include
OMP          = -Xcompiler -fopenmp
OMPI         = -I${OMPI_DIR} -lmpi
BLAS         = -L${BLAS_LIB_DIR} -llibblas.a

LAPACK_LIB_INCL = ${LAPACK_LIB_DIR}/liblapack.a -l${FCC}
LAPACK_C_INCL = ${LAPACK_CWRAPPER}/liblapack_cwrapper.a -l${FCC}
LAPACK = ${LAPACK_LIB_INCL} ${LAPACK_C_INCL} ${LAPACK_ARGS}

CUDA_FLAGS = -lcuda -lcudart
CUDA_INCL  = -L${CUDA_LIB_DIR} ${CUDA_FLAGS}

HOST_FLAGS   = ${LOCAL} ${LAPACK} ${OMP} ${OMPI} ${CUDA}
DEVICE_FLAGS = -arch=${CUDA_ARCH} ${CUDA_INCL}

# - SRC ----------------------------------------------------------------

DEVICE_SRC   = $(wildcard src/*.cu)
HOST_SRC     = $(wildcard src/*.c src/test/*.c)

# - Make ---------------------------------------------------------------

all: build clean

build: gpu cpu
	$(CC) -o $(EXEC) *.o $(HOST_FLAGS)
cpu:
	$(CC) -c $(HOST_FLAGS) $(HOST_SRC)

gpu:
	$(CC) -c $(DEVICE_FLAGS) $(DEVICE_SRC)

clean:
	rm *.o
