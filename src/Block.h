#ifndef _BLOCK_H
#define _BLOCK_H

#define BLK_LEN (16)
#define BLK_SIZE (256)

#define GET_BLK_POS(i, j) (((j) * BLK_LEN) + (i))

typedef double *Block;

int
Block_init(Block blk);

int
Block_init_zero(Block blk);

int
Block_get_elem(Block blk, int i, int j, double *data);

#endif
