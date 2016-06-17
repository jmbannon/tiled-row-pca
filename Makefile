ALL        = row-tile-pca
CC         = mpicc
MAC_FLAGS  = -framework Accelerate
SRC        = $(wildcard *.c)

${ALL}: ${SRC}
		${CC} ${EXECS} -o $@ $^ ${MAC_FLAGS}
