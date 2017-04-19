#ifndef _VECTOR_H
#define _VECTOR_H

typedef struct _Vector {
    int nr_elems;
    int nr_blk_elems;
    double *data;
    double *data_d;
} Vector;

int
Vector_init(Vector *vec,
            int nr_elements);

int
Vector_init_zero(Vector *vec,
                 int nr_elements);

double*
Vector_get_block(Vector *vec,
                 int blk_nr);


void
Vector_print_blocks(Vector *vec);

void
Vector_free(Vector *vec);

#endif
