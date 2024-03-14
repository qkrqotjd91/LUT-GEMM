
#pragma ones

#ifndef KERNELS_H
#define KERNELS_H

#include "nQWeight_fp16.h"

namespace lutGEMM{

void matmul(void* output, nQWeight_fp16 &nqW, void* input, int n, int algo=0);
void matmul(void* output, void* input, nQWeight_fp16 &nqW, int m, int algo=0);
void matmul_gptq(
    int m, int n, int k, void *scale, void *bias,
    void *A, void *B, void *C);
void matmul_gptq_faster(
    int m, int n, int k, void *scale, void *bias,
    void *A, void *B, void *C);
}

#endif

