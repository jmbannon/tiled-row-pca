#include "DistBlockMatrix.h"
#include "BlockMatrix.h"
#include "Block.h"
#include "error.h"
#include "constants.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <mpi.h>

// int
// DistBlockMatrix_to_device(DistBlockMatrix *in, 
//                           double **d_in)
// {
//     return 
//     int nrBlks = in->nr_blk_rows * in->nr_blk_cols;

//     const int inSize = nrBlks * BLK_SIZE * sizeof(double);
//     cudaMalloc((void **)d_in, inSize);

//     cudaMemcpy(*d_in, in->data, inSize, cudaMemcpyHostToDevice);

//     return 0;
// }