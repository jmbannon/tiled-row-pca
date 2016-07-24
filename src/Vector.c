#include "Vector.h"
#include "Block.h"
#include "error.h"
#include "constants.h"
#include <stdlib.h>
#include <stdio.h>

double*
Vector_get_block(Vector *vec,
                 int blk_nr)
{
    if (blk_nr < 0 || blk_nr >= vec->nr_blk_elems) {
        return NULL;
    } else {
        return &vec->data[blk_nr * BLK_LEN];
    }
}

int
Vector_init(Vector *vec,
            int nr_elements)
{
    vec->nr_elems = nr_elements;
    vec->nr_blk_elems = nr_elements / BLK_LEN + (nr_elements % BLK_LEN != 0);
    vec->data = malloc(vec->nr_blk_elems * BLK_LEN * sizeof(double));
    CHECK_MALLOC_RETURN(vec->data);
    return 0;
}

int
Vector_init_zero(Vector *vec,
                 int nr_elements)
{
    vec->nr_elems = nr_elements;
    vec->nr_blk_elems = nr_elements / BLK_LEN + (nr_elements % BLK_LEN != 0);
    vec->data = calloc(vec->nr_blk_elems * BLK_LEN, sizeof(double));
    CHECK_MALLOC_RETURN(vec->data);
    return 0;
}

void
Vector_print_blocks(Vector *vec)
{
    const int max_width = 7;
    for (int i = 0; i < vec->nr_elems; i++) {
        printf("%*.3f ", max_width, vec->data[i]);
        if (i % BLK_LEN == BLK_LEN - 1) {
            printf("  ");
        }
    }
    printf("\n");
}

void
Vector_free(Vector *vec)
{
    free(vec->data);
}
