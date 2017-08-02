#ifndef __TEST_H__
#define __TEST_H__

int Test_BlockMatrix_column_sums();
int Test_DistBlockMatrix_normalize();
int Test_BlockMatrixVector_sub();
int Test_BlockQROperationHouse();
int Test_Matrix_copy_device_to_host();
int Test_TileQR_dgeqt2();

// Single-thread tile qr
int Test_TileQR_17_21_st();
int Test_TileQR_1027_67_st();
int Test_TileQR_67_1027_st();
int Test_TileQR_71_29_st();

// Multi-thread tile qr
int Test_TileQR_17_21_mt();
int Test_TileQR_1027_67_mt();
int Test_TileQR_67_1027_mt();
int Test_TileQR_71_29_mt();
int Test_TileQR_20k_4k_mt();

int TestAll();

#endif