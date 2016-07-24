#ifndef _DOUBLE_BLOCK_H_
#define _DOUBLE_BLOCK_H_

int
DoubleBlock_init(double **dbl_blk);

int
DoubleBlock_init_diag(double **dbl_blk);

int
DoubleBlock_init_rbind(double **rbind, double *top, double *bot);


#endif
