
#ifndef N_Q_WEIGHT_FP16_H
#define N_Q_WEIGHT_FP16_H

namespace lutGEMM{

class nQWeight_fp16;

void dequantize_gpu(nQWeight_fp16 &nqw, void *d_fW, int algo=0);
void dequantize_cpu(nQWeight_fp16 &nqw, void *fW);

class nQWeight_fp16{
public:
    unsigned int* bWeight;  // Weight[kSize/32][nb][mSize]   
    void* alpha;     //  alpha[num_alpha_groups][nb][mSize]
    void* q_bias;   //q_bias[num_alpha_groups][mSize]
    int num_groups;
    int group_size;
    int mSize;
    int kSize;   
    int nb;
    bool is_row_wise_quantize;
    nQWeight_fp16() {}

    /* uint32 bW[kSize/32][nb][mSize]  alpha[num_alpha_groups][mSize][nb] */
    nQWeight_fp16(unsigned int *bW, float *A, int row, int col, int num_bits, 
        bool is_row_wise_quantize, int num_alpha_groups=1, float* q_bias=nullptr){
        parsing(bW, A, row, col, num_bits, is_row_wise_quantize, num_alpha_groups, q_bias);
    }

    void parsing(unsigned int *bW, float *A, int row, int col, int num_bits, 
        bool is_row_wise_quantize, int num_alpha_groups=1, float* q_bias=nullptr);

    ~nQWeight_fp16();
    
    void* getDequantiedWeight(bool onGPU=true);
};

}
#endif // N_Q_WEIGHT_FP16_H
