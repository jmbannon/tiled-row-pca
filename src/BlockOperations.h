#include "constants.h"

#ifndef BLOCK_OPERATIONS_H_
#define BLOCK_OPERATIONS_H_

#include "Block.h"

void
Block_col_sums(Numeric *block,
               Numeric *vec);

void
Block_sub_vec(Numeric *block,
              Numeric *vec);

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
Block_DGEQT2(Numeric *A,
             Numeric *T1);

int
Block_DLARFB(Numeric *A,
             Numeric *VR,
             Numeric *T1);

int
Block_DTSQT2(Numeric *VRin,
             Numeric *T1inout,
             Numeric **VRout);

int
Block_DSSRFB3(Numeric *A_kn,
              Numeric *A_mn,
              Numeric *V_mk,
              Numeric *T1_mk);

#endif
