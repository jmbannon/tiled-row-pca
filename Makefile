EXECS=test
MPICC?=mpicc
OPENBLAS=/usr/local/opt/openblas/include/

all: ${EXECS}

test: DistBlockMatrix.o DistBlockMatrixOperations.o BlockMatrix.o BlockMatrixOperations.o Vector.o
		${MPICC} -o ${EXECS} BlockMatrix.o BlockMatrixOperations.o DistBlockMatrix.o DistBlockMatrixOperations.o Vector.o test.c 
		rm *.o

DistBlockMatrix.o: DistBlockMatrix.c DistBlockMatrix.h error.h
		${MPICC} -c DistBlockMatrix.c

BlockMatrix.o: BlockMatrix.c BlockMatrix.h
		${MPICC} -c BlockMatrix.c -I ${OPENBLAS}

Vector.o: Vector.c Vector.h
		${MPICC} -c Vector.c -I ${OPENBLAS}

DistBlockMatrixOperations.o: DistBlockMatrixOperations.c DistBlockMatrixOperations.h error.h
		${MPICC} -c DistBlockMatrixOperations.c

BlockMatrixOperations.o: BlockMatrixOperations.c BlockMatrixOperations.h
		${MPICC} -c BlockMatrixOperations.c -I ${OPENBLAS}
clean:
		rm ${EXECS}
