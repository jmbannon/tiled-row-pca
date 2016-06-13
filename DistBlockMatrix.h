#include "BlockMatrix.h"

#ifndef MATRIX_H_
#define MATRIX_H_



/* Distributed Tiled-Row wise matrix. */
typedef struct _DistBlockMatrix {
    int nr_nodes;    // Number of nodes
   
    int *node_row_count; // Array where each index represents a node
                         // and the value indicates the number of rows it contains.

    int *node_row_start; // Array where each index represents a node
                         // and the value indicates the starting row # it contains.

    BlockMatrix global;
    BlockMatrix local;
} DistBlockMatrix;

/*
 *  Creates a distributed matrix of all zeroes.
 */
int
DistBlockMatrix_init_zero(DistBlockMatrix *mat,
                          int nr_rows,
                          int nr_cols,
                          int nr_nodes,
                          int curr_node);

int
DistBlockMatrix_seq(DistBlockMatrix *mat,
                    int curr_node);

int
DistBlockMatrix_column_means(DistBlockMatrix *mat,
                             Vector *col_means);

/*
 *  Frees a distributed matrix from memory.
 */
int
DistBlockMatrix_free(DistBlockMatrix *mat,
                     int curr_node);


#endif
