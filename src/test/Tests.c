#include "Tests.h"
#include <stdio.h>

#define MAX_TEST_NAME 40

#define MAX_NUM_TESTS 100


typedef struct _Test {
	char *name;
	int (*testFunction)();
} Test;

Test tests[MAX_NUM_TESTS];
int testCount = 0;

void addTest(char *name, 
			 int (*testFunction)())
{
	Test test = { .name = name, .testFunction = testFunction };
	tests[testCount++] = test;
}

void addAllTests() {
	// addTest("Test_BlockMatrix_column_sums", Test_BlockMatrix_column_sums);
	// addTest("Test_DistBlockMatrix_global_normalize", Test_DistBlockMatrix_normalize);
	// addTest("Test_BlockMatrixVector_sub", Test_BlockMatrixVector_sub);
	// addTest("Test_Matrix_copy_device_to_host", Test_Matrix_copy_device_to_host);
	// addTest("Test_BlockQROperationHouse", Test_BlockQROperationHouse);
	// addTest("Test_TileQR_dgeqt2", Test_TileQR_dgeqt2);

	// addTest("Test_TileQR_17_21_st", Test_TileQR_17_21_st);
	// addTest("Test_TileQR_1027_67_st", Test_TileQR_1027_67_st);
	// addTest("Test_TileQR_67_1027_st", Test_TileQR_67_1027_st);
	// addTest("Test_TileQR_71_29_st", Test_TileQR_71_29_st);

	addTest("Test_TileQR_17_21_mt", Test_TileQR_17_21_mt);
	addTest("Test_TileQR_1027_67_mt", Test_TileQR_1027_67_mt);
	addTest("Test_TileQR_67_1027_mt", Test_TileQR_67_1027_mt);
	addTest("Test_TileQR_71_29_mt", Test_TileQR_71_29_mt);
}

int TestAll() {
	addAllTests();

	int res = 0;
	for (int i = 0; i < testCount; i++) {
		res = (*tests[i].testFunction) ();
		if (res != 0) {
			printf("FAILURE  -  %s\n", tests[i].name);
		} else {
			printf("SUCCESS  -  %s\n", tests[i].name);
		}
	}
	printf("\n");
	return 0;
}