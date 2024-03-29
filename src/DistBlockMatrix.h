#include "BlockMatrix.h"

#ifndef DIST_BLOCK_MATRIX_H_
#define DIST_BLOCK_MATRIX_H_

/** Distributed Tiled-Row wise matrix. */
typedef struct _DistBlockMatrix {
    int nr_nodes;    // Number of nodes
   
    int *node_row_count; // Array where each index represents a node.
                         // and the value indicates the number of rows it contains.

    int *node_row_start; // Array where each index represents a node.
                         // and the value indicates the starting row # it contains.

    BlockMatrix global;  // Global matrix meta info.
    BlockMatrix local;   // Local matrix that contains portions of global on each node.
} DistBlockMatrix;

/**
 * Creates a distributed matrix of all zeroes.
 */
int
DistBlockMatrix_init_zero(DistBlockMatrix *mat,
                          int nr_rows,
                          int nr_cols,
                          int nr_nodes,
                          int curr_node);

/**
 * Creates a distributed matrix of sequential numbers from 0 to (m * n).
 */
int
DistBlockMatrix_seq(DistBlockMatrix *mat,
                    int curr_node);


/*
 * Frees a distributed matrix from memory.
 */
int
DistBlockMatrix_free(DistBlockMatrix *mat,
                     int curr_node);

/**
 * Prints the local portion of a distributed matrix.
 */ 
void
DistBlockMatrix_print_blocks(DistBlockMatrix *mat,
                             int curr_node);

/**
 * Copies local data from host to device.
 */
int
DistBlockMatrix_copy_host_to_device(DistBlockMatrix *in);

int
DistBlockMatrix_copy_device_to_host(DistBlockMatrix *in);

/**
 * CudaMalloc device local matrix.
 */
int
DistBlockMatrix_init_device(DistBlockMatrix *in);


int
DistBlockMatrix_free_device(DistBlockMatrix *in);

#endif
