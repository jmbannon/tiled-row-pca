#include "Matrix.h"
#include "BlockMatrix.h"
#include "constants.h"
#include "error.h"
#include <stdio.h>

/**
 * Initializes a matrix's meta-info but does not allocate
 * memory for its data.
 */
#ifdef __cplusplus
extern "C"
#endif
int
Matrix_init_info(Matrix *mat,
                 int nr_rows,
                 int nr_cols)
{
    mat->nr_rows = nr_rows;
    mat->nr_cols = nr_cols;
    return 0;
}

#ifdef __cplusplus
extern "C"
#endif
int
Matrix_size_bytes(Matrix *mat)
{
    return mat->nr_rows * mat->nr_cols * sizeof(Numeric);
}

int
Matrix_init(Matrix *mat,
	        int nr_rows,
	        int nr_cols)
{
	Matrix_init_info(mat, nr_rows, nr_cols);
    mat->data = (Numeric *)malloc(Matrix_size_bytes(mat));
    CHECK_MALLOC_RETURN(mat->data);

    return 0;
}

int
Matrix_copy_BlockMatrix(Matrix *m, BlockMatrix *cp)
{
    CHECK_ERROR_RETURN(m->nr_rows != cp->nr_rows || m->nr_cols != cp->nr_cols, "Dimensions do not match", INVALID_DIMS);

    for (int i = 0; i < m->nr_rows; i++) {
        for (int j = 0; j < m->nr_cols; j++) {
            m->data[MAT_POS(i, j, m->nr_rows)] = cp->data[POS(i, j, cp->nr_blk_cols)];
        }
    }

    return 0;
}

int Matrix_print(Matrix *mat)
{
	const int max_width = 7;
    int idx;
    for (int i = 0; i < mat->nr_rows; i++) {
        for (int j = 0; j < mat->nr_cols; j++) {
            idx = MAT_POS(i, j, mat->nr_rows);
            printf("%*.3f ", max_width, mat->data[idx]);
        }
        printf("\n");
    }
}

int
Matrix_init_constant(Matrix *mat,
	                 int nr_rows,
	                 int nr_cols,
	                 Numeric constant)
{
	int res = Matrix_init(mat, nr_rows, nr_cols);
    CHECK_ZERO_RETURN(res);

    int size = nr_rows * nr_cols;
    for (int i = 0; i < size; i++) {
        mat->data[i] = constant;
    }

    return 0;
}

int
Matrix_init_diag(Matrix *mat,
	             int nr_rows,
	             int nr_cols,
	             Numeric constant)
{
	int res = Matrix_init_zero(mat, nr_rows, nr_cols);
	CHECK_ZERO_RETURN(res);

	int min = nr_rows < nr_cols ? nr_rows : nr_cols;
	for (int i = 0; i < min; i++) {
		mat->data[MAT_POS(i, i, mat->nr_rows)] = constant;
	}

	return 0;
}

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
                 int nr_cols)
{
	Matrix_init_info(mat, nr_rows, nr_cols);
    int size = mat->nr_rows * mat->nr_cols;

    mat->data = (Numeric *)calloc(size, sizeof(Numeric));
    CHECK_MALLOC_RETURN(mat->data);

    return 0;
}

int
Matrix_free(Matrix *mat)
{
	if (mat->data != NULL) {
        free(mat->data);
    }
    return 0;
}
