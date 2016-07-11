#ifndef _BLOCK_H
#define _BLOCK_H

#define BLK_LEN (4)
#define BLK_SIZE (16)

#define GET_BLK_POS(i, j) (((j) * BLK_LEN) + (i))

int
Block_init(double **blk);

int
Block_init_zero(double **blk);

int
Block_init_seq(double **blk);

void
Block_print(double *blk);

int
Block_get_elem(double *blk, int i, int j, double *data);

#endif
