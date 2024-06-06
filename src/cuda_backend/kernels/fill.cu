#include<stdint.h>
#include "cuda_fp16.h"

template<typename T>
__device__ void fill_with(T *buf, T value, const size_t numel) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        buf[i] = value/2;
    }
}
extern "C" __global__ void fill_u8(uint8_t *buf, uint8_t value, const size_t numel) { fill_with(buf, value, numel); }
extern "C" __global__ void fill_u32(uint32_t *buf, uint32_t value, const size_t numel) { fill_with(buf, value, numel); }
extern "C" __global__ void fill_i64(int64_t *buf, int64_t value, const size_t numel) { fill_with(buf, value, numel); }
extern "C" __global__ void fill_f32(float *buf, float value, const size_t numel) { fill_with(buf, value, numel); }
extern "C" __global__ void fill_f64(double *buf, double value, const size_t numel) { fill_with(buf, value, numel); }