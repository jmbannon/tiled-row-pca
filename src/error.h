#ifndef ERROR_H_
#define ERROR_H_

#define CHECK_ZERO_RETURN(res)\
if ((res) != 0) \
{\
    return (res);\
}

#define CHECK_MALLOC_RETURN(res)\
if ((res) == NULL) \
{\
    return MALLOC_FAIL;\
}

#define MALLOC_FAIL -1
#define INVALID_DIMS -1000
#define INVALID_NODES -1001
#define INVALID_INDICES -1002

#endif
