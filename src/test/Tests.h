#ifndef __TEST_H__
#define __TEST_H__

int Test_BlockMatrix_column_sums();
int Test_DistBlockMatrix_normalize();
int Test_BlockMatrixVector_sub();
int Test_BlockQROperationHouse();
int Test_Matrix_copy_device_to_host();
int Test_TileQR_dgeqt2();
int Test_TileQR_dgeqt2_rect();
int Test_TileQR_16_16();
int Test_TileQR_1024_64();

int TestAll();

#endif