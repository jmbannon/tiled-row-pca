EXECS=test
MPICC?=mpicc
APPLE_LAPACK=framework Accelerate
all: ${EXECS}

test: DistBlockMatrix.o DistBlockMatrixOperations.o BlockMatrix.o BlockMatrixOperations.o Vector.o lapacke.h
		${MPICC} -o ${EXECS} BlockMatrix.o BlockMatrixOperations.o DistBlockMatrix.o DistBlockMatrixOperations.o Vector.o test.c -${APPLE_LAPACK} 
		rm *.o

DistBlockMatrix.o: DistBlockMatrix.c DistBlockMatrix.h error.h
		${MPICC} -c DistBlockMatrix.c

BlockMatrix.o: BlockMatrix.c BlockMatrix.h lapacke.h
		${MPICC} -c BlockMatrix.c

Vector.o: Vector.c Vector.h
		${MPICC} -c Vector.c

DistBlockMatrixOperations.o: DistBlockMatrixOperations.c DistBlockMatrixOperations.h error.h
		${MPICC} -c DistBlockMatrixOperations.c

BlockMatrixOperations.o: BlockMatrixOperations.c BlockMatrixOperations.h
		${MPICC} -c BlockMatrixOperations.c
clean:
		rm *.o
		rm ./${EXECS}

