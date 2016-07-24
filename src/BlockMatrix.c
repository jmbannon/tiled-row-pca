#include "BlockMatrix.h"
#include "Block.h"
#include "error.h"
#include "Vector.h"
#include <stdbool.h>
#include "lapacke.h"
#include "constants.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * Initializes a block matrix. Boolean param specifying whether to
 * actually allocate memory for it.
 *
 * @param mat Matrix to initialize.
 * @param nr_rows Number of rows.
 * @param nr_cols Number of columns.
 * @param init_data Boolean specifying whether to allocate memory
 *                  for matrix data.
 */
static int
BlockMatrix_init_zero_flag(BlockMatrix *mat,
                           int nr_rows,
                           int nr_cols,
                           bool init_data)
{
    mat->nr_rows = nr_rows;
    mat->nr_cols = nr_cols;
    mat->nr_blk_rows = nr_rows / BLK_LEN + (nr_rows % BLK_LEN != 0);
    mat->nr_blk_cols = nr_cols / BLK_LEN + (nr_cols % BLK_LEN != 0);
    if (init_data) {
        int nr_elements = mat->nr_blk_rows * mat->nr_blk_cols * BLK_SIZE;
        mat->data = (double *)calloc(nr_elements, sizeof(double));
        CHECK_MALLOC_RETURN(mat->data);
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
BlockMatrix_init_zero(BlockMatrix *mat,
                      int nr_rows,
                      int nr_cols)
{
    return BlockMatrix_init_zero_flag(mat, nr_rows, nr_cols, true);
}

/**
 * Initializes a matrix's meta-info but does not allocate
 * memory for its data.
 *
 * @param mat Matrix to initialize info.
 * @paran nr_rows Number of rows.
 * @param nr_cols Number of columns.
 */
int
BlockMatrix_init_info(BlockMatrix *mat,
                      int nr_rows,
                      int nr_cols)
{
    return BlockMatrix_init_zero_flag(mat, nr_rows, nr_cols, false);
}

int
BlockMatrix_print(BlockMatrix *mat)
{
    int idx;
    int i, j;
    for (i = 0; i < mat->nr_rows; i++) {
        for (j = 0; j < mat->nr_cols; j++) {
            idx = POS(i,j,mat->nr_blk_cols);
            printf("%.3f ", mat->data[idx]);
        }
        printf(" %d %d \n", i, j);
    }
    return 0;
}

void
BlockMatrix_print_blocks(BlockMatrix *mat)
{
    const int max_width = 7;
    int idx;
    for (int i = 0; i < mat->nr_rows; i++) {
        for (int j = 0; j < mat->nr_cols; j++) {
            idx = POS(i, j, mat->nr_blk_cols);
            printf("%*.3f ", max_width, mat->data[idx]);
            if (j % BLK_LEN == (BLK_LEN - 1)) {
                printf("  ");
            }
        }
        printf("\n");
        if (i % BLK_LEN == (BLK_LEN - 1)) {
            printf("\n");
        }
    }
}

void
BlockMatrix_print_padding(BlockMatrix *mat)
{
    int idx;
    int i, j;
    for (i = 0; i < mat->nr_blk_rows * BLK_LEN; i++) {
        for (j = 0; j < mat->nr_blk_cols * BLK_LEN; j++) {
            idx = POS(i,j,mat->nr_blk_cols);
            printf("%.3f ", mat->data[idx]);
        }
        printf(" %d %d \n", i, j);
    }
}

int test_1()
{
    // note, to understand this part take a look in the MAN pages, at section of parameters.
    char    TRANS = 'N';
    int     INFO=3;
    int     LDA = 3;
    int     LDB = 3;
    int     N = 3;
    int     NRHS = 1;
    int     IPIV[3] ;
 
    double  A[9] =
    {
    1, 2, 3,
    2, 3, 4,
    3, 4, 1
    };
 
    double B[3] =
    {
    -4,
    -1,
    -2
    };
// end of declarations
 
    //void LAPACK_dgetrf( lapack_int* m, lapack_int* n, double* a, lapack_int* lda, lapack_int* ipiv, lapack_int *info );
    LAPACK_dgetrf(&N,&N,A,&LDA,IPIV,&INFO);
    return 0;
}
