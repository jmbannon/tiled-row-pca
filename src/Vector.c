#include "Vector.h"
#include "Block.h"
#include "error.h"
#include "constants.h"
#include <stdlib.h>
#include <stdio.h>

void
Vector_set_dimensions(Vector *vec, int nr_elements)
{
    vec->nr_elems = nr_elements;
    vec->nr_blk_elems = nr_elements / BLK_LEN + (nr_elements % BLK_LEN != 0);
    vec->data = NULL;
    vec->data_d = NULL;
}

int
Vector_size_bytes(Vector *vec)
{
    return vec->nr_blk_elems * BLK_LEN * sizeof(Numeric);
}

Numeric*
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
    Vector_set_dimensions(vec, nr_elements);
    vec->data = malloc(vec->nr_blk_elems * BLK_LEN * sizeof(Numeric));
    CHECK_MALLOC_RETURN(vec->data);
    return 0;
}

int
Vector_init_constant(Vector *vec,
                     int nr_elements,
                     Numeric constant)
{
    int res = Vector_init(vec, nr_elements);
    CHECK_ZERO_RETURN(res);

    for (int i = 0; i < nr_elements; i++) {
        vec->data[i] = constant;
    }
    return 0;
}

int
Vector_init_zero(Vector *vec,
                 int nr_elements)
{
    Vector_set_dimensions(vec, nr_elements);
    vec->data = calloc(vec->nr_blk_elems * BLK_LEN, sizeof(Numeric));
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
