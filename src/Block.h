#ifndef _BLOCK_H
#define _BLOCK_H

#include <stdbool.h>

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

int
Block_zero_tri(double *blk, bool upper, bool diag);

void
Block_print_rbind(double *rbind);

#endif
