#ifndef TMP_WEIGHT_HPP
#define TMP_WEIGHT_HPP


class tmpWeight{
public:
    static tmpWeight& getInstance(){
        static tmpWeight ins;
        return ins;
    }

    float* getWeight(int Size){
        if(Size > size){
            mem_free();
            size = Size;
            cudaMallocManaged(&mem, sizeof(float) * size);
        }
        return mem;
    }

private:
    void mem_free(){
        if(mem != nullptr)
            cudaFree(mem);
    }
    float *mem = nullptr;
    int size = 0;

    tmpWeight(const tmpWeight&) = delete;
    tmpWeight& operator=(const tmpWeight&) = delete;
    tmpWeight(/* args */){ }
    ~tmpWeight(){
        mem_free();
    }
};


#endif // TMP_WEIGHT_HPP