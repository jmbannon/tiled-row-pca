EXECS=test
MPICC?=mpicc
OPENBLAS=/usr/local/opt/openblas/include/

all: ${EXECS}

test: DistBlockMatrix.o BlockMatrix.o Vector.o
		${MPICC} -o ${EXECS} BlockMatrix.o DistBlockMatrix.o Vector.o test.c 
		rm *.o

DistBlockMatrix.o: DistBlockMatrix.c DistBlockMatrix.h error.h
		${MPICC} -c DistBlockMatrix.c

BlockMatrix.o: BlockMatrix.c BlockMatrix.h
		${MPICC} -c BlockMatrix.c -I ${OPENBLAS}

Vector.o: Vector.c Vector.h
		${MPICC} -c Vector.c -I ${OPENBLAS}

clean:
		rm ${EXECS}
