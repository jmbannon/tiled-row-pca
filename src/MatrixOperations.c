#include "MatrixOperations.h"
#include "Matrix.h"
#include "lapacke.h"
#include "error.h"
#include <stdio.h>
#include <stdlib.h>

/** Performs LAPACK QR decomposition on a matrix. */
int Matrix_qr_r(Matrix *m) {

	int res;

	Numeric work_size_n;
    Numeric *work;
    Numeric *tau;

    int work_size;
    int work_query = -1; // Used to get optimal work size.
    int min_dim = m->nr_rows > m->nr_cols ? m->nr_cols : m->nr_rows;

    tau = malloc(min_dim * sizeof(Numeric));
    CHECK_MALLOC_RETURN(tau);

    #if FLOAT_NUMERIC
        LAPACK_sgeqrf(&m->nr_rows, &m->nr_cols, m->data, &m->nr_rows, tau, &work_size_n, &work_query, &res);
    #else
        LAPACK_dgeqrf(&m->nr_rows, &m->nr_cols, m->data, &m->nr_rows, tau, &work_size_n, &work_query, &res);
    #endif
    CHECK_ZERO_ERROR_RETURN(res, "Failed to query size of array");

    work_size = (int)work_size_n;
    work = malloc(work_size * sizeof(Numeric));
    CHECK_MALLOC_RETURN(work);

    #if FLOAT_NUMERIC
        LAPACK_sgeqrf(&m->nr_rows, &m->nr_cols, m->data, &m->nr_rows, tau, work, &work_size, &res);
    #else
        LAPACK_dgeqrf(&m->nr_rows, &m->nr_cols, m->data, &m->nr_rows, tau, work, &work_size, &res);
    #endif
    CHECK_ZERO_ERROR_RETURN(res, "Failed to compute lapack QR decomposition");

    free(tau);
    free(work);

    return 0;
}