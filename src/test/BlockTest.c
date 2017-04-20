#include <stdio.h>
#include <stdlib.h>
#include "BlockTest.h"
#include "../error.h"
#include "../Block.h"
#include "../DoubleBlock.h"
#include "../BlockOperations.h"
#include "../constants.h"

int test_DGEQT2()
{
    int res;
    Numeric *seq, *seqR, *seqT;

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

int test_DGEQT3()
{
    int res;
    Numeric *seq, *seqR, *seqT, *rbind;

    res = Block_init_seq(&seq);
    CHECK_ZERO_RETURN(res);
    res = Block_init_seq(&seqT);
    CHECK_ZERO_RETURN(res);

    res = Block_DTSQT2(seq, seqT, &rbind);
    CHECK_ZERO_RETURN(res);

    printf("R and Householder Vectors:\n");
    DoubleBlock_print(rbind);
    printf("T Matrix:\n");
    Block_print(seqT);

    return 0;
}

int test_Block_tri()
{
    int res;
    Numeric *seq;

    for (int upper = 0; upper <= 1; upper++) {
        for (int diag = 0; diag <= 1; diag++) {
            res = Block_init_seq(&seq);
            CHECK_ZERO_RETURN(res);
            printf("upper=%d, diag=%d\n", upper, diag);
            res = Block_zero_tri(seq, upper, diag);
            CHECK_ZERO_RETURN(res);
            Block_print(seq);
            free(seq);
        }
    }
    return 0;
}

int test_Block_init_rbind()
{
    int res;
    Numeric *seq1, *seq2, *rbind;
    res = Block_init_seq(&seq1);
    CHECK_ZERO_RETURN(res);
    res = Block_init_seq(&seq2);
    CHECK_ZERO_RETURN(res);
    res = DoubleBlock_init_rbind(&rbind, seq1, seq2);
    CHECK_ZERO_RETURN(res);

    printf("two seq's rbinded\n");
    DoubleBlock_print(rbind);
    free(seq1);
    free(seq2);
    free(rbind);
}
