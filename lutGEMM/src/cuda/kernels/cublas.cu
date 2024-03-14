
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace kernel{

template <typename T, typename S>
inline cublasStatus_t cublas_gemm_ex(T *A,  T *B,  S *C,
                                    int m, int n, int k);

typedef cublasStatus_t<__half

template <typename T, typename S>
inline cublasStatus_t cublas_gemm_ex(T *A,  T *B,  S *C,
                                    int m, int n, int k) {
    static S alpha = 1;
    static S beta  = 0;
    static cublasHandle_t handle = nullptr;
    if(handle == nullptr) cublasCreate(&handle);
    
    cudaDataType_t AType, BType, CType;
    cublasComputeType_t  ComputeType;
    if (std::is_same<T, float>::value) {
        AType = BType = CType = CUDA_R_32F;
        ComputeType = CUBLAS_COMPUTE_32F_FAST_TF32;
    } else if (std::is_same<T, __half>::value) {
        AType = BType = CType = CUDA_R_16F;
        ComputeType = CUBLAS_COMPUTE_16F;
    } else if (std::is_same<T, int8_t>::value) {
        AType = BType = CUDA_R_8I;
        CType = CUDA_R_32I;
        ComputeType = CUBLAS_COMPUTE_32I;
    } else {
        printf("Not supported data type.");
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    return cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                          n, m, k, 
                          &alpha,
                          B, BType, n,
                          A, AType, k,
                          &beta,
                          C, CType, n,
                          ComputeType,
                          CUBLAS_GEMM_DFALT);
}

}
