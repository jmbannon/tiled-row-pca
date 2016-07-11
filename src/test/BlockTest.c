#include <stdio.h>
#include "BlockTest.h"
#include "../error.h"
#include "../Block.h"
#include "../BlockOperations.h"

int test_DGEQT2()
{
    int res;
    double *seq, *seqR, *seqT;

    res = Block_init_seq(&seq);
    CHECK_ZERO_RETURN(res);
    res = Block_init_seq(&seqR);
    CHECK_ZERO_RETURN(res);
    res = Block_init(&seqT);
    CHECK_ZERO_RETURN(res);
    
    res = Block_DGEQT2(seqR, seqT);
    CHECK_ZERO_RETURN(res);

    res = Block_DLARFB(seq, seqR, seqT);
    CHECK_ZERO_RETURN(res);

    printf("R and Householder Vectors:\n");
    Block_print(seqR);
    printf("T Matrix:\n");
    Block_print(seqT);

    printf("DLARFB Matrix:\n");
    Block_print(seq);

    return 0;
}
