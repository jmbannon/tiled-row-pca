#ifndef BLOCK_OPERATIONS_H_
#define BLOCK_OPERATIONS_H_

#include "Block.h"

void
Block_col_sums(double *block,
               double *vec);

void
Block_sub_vec(double *block,
              double *vec);

/**
 * Helper function for QR.R
 * Uses the name DGEQT2 based on the Fortran function used in most implementations of tileQR. \cr
 *
 * Performs QR decomposition on a diagonal tile. Computes matrix R for A = QR \cr
 * and matrix T for Q = I + Y %*% T %*% t(Y) using the householder vectors.
 * 
 * @param A Diagonal tile of a matrix.
 * @return RV R for A = QR
 * @return T1 T for Q = I + Y %*% T %*% t(Y)
 */ 
int
Block_DGEQT2(double *A,
             double *T1);

int
Block_DLARFB(double *A,
             double *VR,
             double *T1);

int
Block_DTSQT2(double *VRin,
             double *T1inout,
             double **VRout);

int
Block_DSSRFB3(double *A_kn,
              double *A_mn,
              double *V_mk,
              double *T1_mk);

#endif
