#include "DistBlockMatrix.h"
#include "BlockMatrix.h"
#include "Block.h"
#include "error.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <mpi.h>

/*
 * Partitions the number of rows evenly amongst the nodes.
 *
 * Each index of the two arrays represents a node.
 * node_row_count represents the number of rows on the given node.
 * node_row_start represents the row # of the first row located on that node.
 */
static int
DistBlockMatrix_dist_rows(int *node_row_count,
                          int *node_row_start,
                          int nr_nodes,
                          int nr_rows)
{
    int node_row_offset = 0;
    int i = 0; 
    for (i = 0; i < nr_rows / BLK_LEN; i++) {
        node_row_count[i % nr_nodes] += BLK_LEN;
    }
    node_row_count[i % nr_nodes] += nr_rows % BLK_LEN;
    
    node_row_start[0] = 0;
    for (int i = 1; i < nr_nodes; i++) {
        node_row_offset += node_row_count[i-1]; 
        if (node_row_count[i] != 0) {
            node_row_start[i] = node_row_offset;
        } else {
            node_row_start[i] = 0;
        }
    }

    return 0;
}

/*
 *  Creates a distributed matrix of all zeroes.
 */
int
DistBlockMatrix_init_zero(DistBlockMatrix *mat,
                         int nr_rows,
                         int nr_cols,
                         int nr_nodes,
                         int curr_node)
{
    int res;
    if (nr_rows <= 0 || nr_cols <= 0 || nr_nodes <= 0)
        return -1;

    mat->nr_nodes = nr_nodes;
    
    res = BlockMatrix_init_info(&mat->global, nr_rows, nr_cols);
    CHECK_ZERO_RETURN(res);

    mat->node_row_count = (int *)calloc(nr_nodes, sizeof(int));
    CHECK_MALLOC_RETURN(mat->node_row_count);

    mat->node_row_start = (int *)calloc(nr_nodes, sizeof(int));
    CHECK_MALLOC_RETURN(mat->node_row_start);

    res = DistBlockMatrix_dist_rows(mat->node_row_count, mat->node_row_start, mat->nr_nodes, nr_rows);    
    CHECK_ZERO_RETURN(res);

    res = BlockMatrix_init_zero(&mat->local, mat->node_row_count[curr_node], nr_cols);
    CHECK_ZERO_RETURN(res);
    
    return 0;
}

int
DistBlockMatrix_seq(DistBlockMatrix *mat,
                    int curr_node)
{
    int counter = mat->node_row_start[curr_node] * mat->local.nr_cols;
    int idx, i, j;
    for (i = 0; i < mat->local.nr_rows; i++) {
        for (j = 0; j < mat->local.nr_cols; j++) {
            idx = POS(i,j,mat->local.nr_blk_cols);
            mat->local.data[idx] = (double)counter++;
        }
    }
    return 0;
}

/*
 *  Frees a distributed matrix from memory.
 */
int
DistBlockMatrix_free(DistBlockMatrix *mat,
                     int curr_node)
{
    free(mat->node_row_count);
    free(mat->node_row_start);
    if (curr_node == 0) {
        free(mat->global.data);
    }
    free(mat->local.data);
    return 0;
}

void
DistBlockMatrix_print_blocks(DistBlockMatrix *mat,
                             int curr_node)
{
    for (int i = 0; i < mat->nr_nodes; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == curr_node) {
            printf("rank %d\n", curr_node);
            BlockMatrix_print_blocks(&mat->local);
        }
    }
}
