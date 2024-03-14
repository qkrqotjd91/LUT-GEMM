#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include <memory>

/* submodule */
#include "gtest/gtest.h"

/* custom module */
#include "custom_random.h"
#include "timer.h"

#include <sys/time.h>

#include "lutGEMM"

// #include <cuda.h>
// #include <cuda_fp16.h>
// #include <cuda_runtime.h>
// #include <cuda_fp16.h>
// #include <cublas_v2.h>
// #include <cublasLt.h>

// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"


// #include <sys/time.h>
// #include <cuda_profiler_api.h>
// #include <cublas_v2.h>
// #include <cuda.h>
// #include <cuda_fp16.h>
// #include <cuda_runtime.h>

// #include "lutGEMM"

#ifndef GTEST_PIRNTF
#define GTEST_PIRNTF(...){\
    printf("\033[32m[          ]");\
    printf("\033[0m ");\
    printf(__VA_ARGS__);\
    printf("\n");\
}
#endif