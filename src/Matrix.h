#ifndef _MATRIX_H_
#define _MATRIX_H_

#include "constants.h"

#define MAT_POS(i, j, nr_rows) ((j) * (nr_rows) + (i))

typedef struct _Matrix {
    int nr_rows;   // Number of rows
    int nr_cols;   // Number of columns

    Numeric *data;      // Host data
    Numeric *data_d;    // Device data
} Matrix;

int
Matrix_init(Matrix *mat,
	        int nr_rows,
	        int nr_cols);

int
Matrix_init_constant(Matrix *mat,
	                 int nr_rows,
	                 int nr_cols,
	                 Numeric constant);

int
Matrix_init_diag(Matrix *mat,
	             int nr_rows,
	             int nr_cols,
	             Numeric constant);

/**
 * Initializes a matrix with 0s.
 *
 * @param mat Matrix to initialize.
 * @param nr_rows Number of rows.
 * @param nr_cols Number of columns.
 */
int
Matrix_init_zero(Matrix *mat,
                 int nr_rows,
                 int nr_cols);

int
Matrix_free(Matrix *mat);

/**
 * Initializes a matrix's meta-info but does not allocate
 * memory for its data.
 *
 * @param mat Matrix to initialize info.
 * @paran nr_rows Number of rows.
 * @param nr_cols Number of columns.
 */
#ifdef __cplusplus
extern "C"
#endif
int
Matrix_init_info(Matrix *mat,
                 int nr_rows,
                 int nr_cols);

int
Matrix_print(Matrix *mat);

#ifdef __cplusplus
extern "C"
#endif
int
Matrix_size_bytes(Matrix *mat);

/**
 * Copies data from host to device.
 */
#ifdef __cplusplus
extern "C"
#endif
int
Matrix_copy_host_to_device(Matrix *in);

#ifdef __cplusplus
extern "C"
#endif
int
Matrix_copy_device_to_host(Matrix *in);

/**
 * CudaMalloc device matrix.
 */
#ifdef __cplusplus
extern "C"
#endif
int
Matrix_init_device(Matrix *in,
				   int nr_rows,
				   int nr_cols);

#ifdef __cplusplus
extern "C"
#endif
int
Matrix_free_device(Matrix *in);

#ifdef __cplusplus
extern "C"
#endif
int
Matrix_init_constant_device(Matrix *mat,
	                        int nr_rows,
	                        int nr_cols,
	                        Numeric constant);

#ifdef __cplusplus
extern "C"
#endif
int
Matrix_init_diag_device(Matrix *mat,
			            int nr_rows,
			            int nr_cols,
			            Numeric constant);

#ifdef __cplusplus
extern "C"
#endif
int
Matrix_init_seq_mem_device(Matrix *mat,
			               int nr_rows,
			               int nr_cols);

#ifdef __cplusplus
extern "C"
#endif
int
Matrix_init_zero_device(Matrix *mat,
                 		int nr_rows,
                 		int nr_cols);

#ifdef __cplusplus
extern "C"
#endif
int
Matrix_init_rand_device(Matrix *mat,
			            int nr_rows,
			            int nr_cols,
			            unsigned long seed);

#ifdef __cplusplus
extern "C"
#endif
int
Matrix_add_diag_device(Matrix *mat,
			           Numeric constant);

#endif