EXEC         = row-tile-pca
CC           = mpicc
INCL         = -I./include
LAPACK_INCL  = -framework Accelerate
SRC          = $(wildcard src/*.c)

FLAGS        = ${INCL} ${LAPACK_INCL}
# == Edit to your configuration =======

WORKING_DIR = /Users/jb/workspace/pca2

# ======================================

DEFINES =   -D WORKING_DIR=${WORKING_DIR} \
			-D CC=${CC} \
			

${EXEC}: ${SRC}
		${CC} -o $@ $^ ${FLAGS} ${DEFINES}

