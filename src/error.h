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
#define CHECK_ZERO_ERROR_RETURN(res, error_message) CHECK_ERROR_RETURN((res) != 0, error_message, 0)

#define CHECK_SUCCESS_RETURN(res) CHECK_RETURN((res) != cudaSuccess, 1)
#define CHECK_MALLOC_RETURN(res) CHECK_RETURN((res) == NULL, MALLOC_FAIL)


#define MALLOC_FAIL -1
#define INVALID_DIMS -1000
#define INVALID_NODES -1001
#define INVALID_INDICES -1002

#endif
