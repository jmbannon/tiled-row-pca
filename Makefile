include make.inc

# - CC ----------------------------------------------------------------

EXEC            = row-tile-pca
TEST_EXEC       = test-row-tile-pca
PERF_EXEC       = perf-row-tile-pca
CC              = nvcc -m64
FCC             = gfortran

# - Flags -------------------------------------------------------------

LOCAL           = -I./include
OMP             = -Xcompiler -fopenmp
OMPI            = -I${OMPI_DIR} -lmpi
BLAS            = -L${BLAS_LIB_DIR} -llibblas.a

LAPACK_LIB_INCL = ${LAPACK_LIB_DIR}/liblapack.a -l${FCC}
LAPACK_C_INCL   = ${LAPACK_CWRAPPER}/liblapack_cwrapper.a -l${FCC}
LAPACK          = ${LAPACK_LIB_INCL} ${LAPACK_C_INCL} ${LAPACK_ARGS}

CUDA_FLAGS      = -lcuda -lcudart -lcublas -lcusolver -lcublas_device --use_fast_math 
CUDA_INCL       = -I/usr/local/cuda/include -L${CUDA_LIB_DIR} ${CUDA_FLAGS}

SHARED_FLAGS    = -Wno-deprecated-gpu-targets
HOST_FLAGS      = ${SHARED_FLAGS} ${LOCAL} ${LAPACK} ${OMP} ${OMPI} ${CUDA}
DEVICE_FLAGS    = ${SHARED_FLAGS} -arch=${CUDA_ARCH} ${CUDA_INCL}

# - SRC ----------------------------------------------------------------

PERF_SRC        = $(wildcard src/perf/*.c)
TEST_SRC        = $(wildcard src/test/*.c)
MAIN_SRC        = $(wildcard src/main/*.c)
DEVICE_SRC      = $(wildcard src/kernels/*.cu src/device/*.cu)
HOST_SRC        = $(wildcard src/*.c)

# - Make ---------------------------------------------------------------

main: gpu_compile cpu_compile main_compile
	$(CC) -o $(EXEC) *.o $(HOST_FLAGS) ${DEVICE_FLAGS}
	rm -f *.o

test: gpu_compile cpu_compile test_compile
	$(CC) -o $(TEST_EXEC) *.o $(HOST_FLAGS) ${DEVICE_FLAGS}
	rm -f *.o

perf: gpu_compile cpu_compile perf_compile
	$(CC) -o $(PERF_EXEC) *.o $(HOST_FLAGS) ${DEVICE_FLAGS}
	rm -f *.o

main_compile:
	$(CC) -c ${DEVICE_FLAGS} $(HOST_FLAGS) $(MAIN_SRC)

test_compile:
	$(CC) -c $(HOST_FLAGS) $(TEST_SRC)

perf_compile:
	$(CC) -c $(HOST_FLAGS) $(PERF_SRC)

cpu_compile:
	$(CC) -c $(HOST_FLAGS) $(HOST_SRC)

gpu_compile:
	$(CC) -c $(DEVICE_FLAGS) ${HOST_FLAGS} -dc $(DEVICE_SRC)

clean:
	rm -f *.o
