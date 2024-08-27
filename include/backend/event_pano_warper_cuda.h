#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

bool test_cuda(int N, float *h_A, float *h_B, float *h_C);

__global__ void warpEvents(float3 *h_rotated_ray, float2 *warped_pixel_pose, float2 center, float fx, float fy, int n);
void warpEventsWrapper(float3 *h_rotated_ray, float2 *warped_pixel_pose, float2 center, float fx, float fy, int n);

__global__ void accumulateIL(float *d_IL_old, float *d_IL_new, int IL_old_row, int IL_old_cols, bool* h_oldevent, int2* d_xy, float2 *d_dxdy, int n);
__global__ void getIntAndDecimal(float2 *xx_yy, float2 *dx_dy, int2 *x_y ,int n);
void accumulatePolarityWrapper(float2 *h_new_xx_yy, float2 *h_old_xx_yy, float *h_IL_old, float *h_IL_new, int2 IL_old_dim, int2 IL_new_dim, int num_new_events, int num_old_events);

void getIntAndDecimalWrapper(float2 *h_xx_yy, int2 *h_x_y, float2 *h_dx_dy, int n);

__global__ void matrixVectorMultiply(const float* R, const float* vec, float* result);
void matrixVectorMultWrapper(const float* h_rot_m, const float* h_vec, float* h_res);

