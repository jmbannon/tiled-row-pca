#include <cuda.h>
#include <cuda_runtime.h>

#ifndef ERROR_H_
#define ERROR_H_

#define CHECK_RETURN(error_condition, ret)\
if (error_condition) \
{\
    return (ret);\
}

#define CHECK_ERROR_RETURN(error_condition, error_message, ret)\
if (error_condition) \
{\
    printf("\n ERROR: %s\n", error_message);\
    return (ret);\
}

#define CHECK_ZERO_RETURN(res) CHECK_RETURN((res) != 0, res)
#define CHECK_ZERO_ERROR_RETURN(res, error_message) CHECK_ERROR_RETURN((res) != 0, error_message, res)

#define CHECK_SUCCESS_RETURN(res) CHECK_RETURN((res) != cudaSuccess, 1)
#define CHECK_MALLOC_RETURN(res) CHECK_RETURN((res) == NULL, MALLOC_FAIL)



#define CHECK_CUBLAS_RETURN(res, error)\
switch(res) \
{\
    case CUBLAS_STATUS_SUCCESS: res = 0; break;\
    case CUBLAS_STATUS_NOT_INITIALIZED: printf("\n ERROR: CUBLAS_STATUS_NOT_INITIALIZED\n %s\n", error); return CUBLAS_STATUS_NOT_INITIALIZED;\
    case CUBLAS_STATUS_ALLOC_FAILED: printf("\n ERROR: CUBLAS_STATUS_ALLOC_FAILED\n %s\n", error); return CUBLAS_STATUS_ALLOC_FAILED;\
    case CUBLAS_STATUS_INVALID_VALUE: printf("\n ERROR: CUBLAS_STATUS_INVALID_VALUE\n %s\n", error); return CUBLAS_STATUS_INVALID_VALUE;\
    case CUBLAS_STATUS_ARCH_MISMATCH: printf("\n ERROR: CUBLAS_STATUS_ARCH_MISMATCH\n %s\n", error); return CUBLAS_STATUS_ARCH_MISMATCH;\
    case CUBLAS_STATUS_MAPPING_ERROR: printf("\n ERROR: CUBLAS_STATUS_MAPPING_ERROR\n %s\n", error); return CUBLAS_STATUS_MAPPING_ERROR;\
    case CUBLAS_STATUS_EXECUTION_FAILED: printf("\n ERROR: CUBLAS_STATUS_EXECUTION_FAILED\n %s\n", error); return CUBLAS_STATUS_EXECUTION_FAILED;\
    case CUBLAS_STATUS_INTERNAL_ERROR: printf("\n ERROR: CUBLAS_STATUS_INTERNAL_ERROR\n %s\n", error); return CUBLAS_STATUS_INTERNAL_ERROR;\
    default: printf("\n ERROR: UKNOWN_CUBLAS_ERROR\n %s\n", error); return 1;\
}





#define MALLOC_FAIL -1
#define INVALID_DIMS -1000
#define INVALID_NODES -1001
#define INVALID_INDICES -1002

#endif
