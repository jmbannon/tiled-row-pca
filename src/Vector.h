#include "constants.h"

#ifndef _VECTOR_H
#define _VECTOR_H

typedef struct _Vector {
    int nr_elems;
    int nr_blk_elems;
    Numeric *data;
    Numeric *data_d;
} Vector;

int
Vector_init(Vector *vec,
            int nr_elements);

int
Vector_init_zero(Vector *vec,
                 int nr_elements);

int
Vector_init_constant(Vector *vec,
                     int nr_elements,
                     Numeric constant);

Numeric*
Vector_get_block(Vector *vec,
                 int blk_nr);


void
Vector_print_blocks(Vector *vec);

void
Vector_free(Vector *vec);

#ifdef __cplusplus
extern "C"
#endif
int
Vector_size_bytes(Vector *in);

/**
 * Copies data from host to device.
 */
#ifdef __cplusplus
extern "C"
#endif
int
Vector_copy_host_to_device(Vector *in);

#ifdef __cplusplus
extern "C"
#endif
int
Vector_copy_device_to_host(Vector *in);

/**
 * CudaMalloc device vector.
 */
#ifdef __cplusplus
extern "C"
#endif
int
Vector_init_device(Vector *in);

#ifdef __cplusplus
extern "C"
#endif
int
Vector_free_device(Vector *in);

#endif
