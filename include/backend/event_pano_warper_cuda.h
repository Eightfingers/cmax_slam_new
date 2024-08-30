#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

class CudaEventWarper
{
public:
    CudaEventWarper();
    ~CudaEventWarper();

    void setSubsetSize(int subset_size) { subset_size_ = subset_size; }
    void setILOldNewSize(int IL_old_rows, int IL_old_cols, int IL_new_rows, int IL_new_cols)
    {
        IL_old_rows_ = IL_old_rows;
        IL_old_cols_ = IL_old_cols;
        IL_new_rows_ = IL_new_rows;
        IL_new_cols_ = IL_new_cols;
    }
    void setWarpingParameters(double fx, double fy, double center_x, double center_y)
    {
        fx_ = fx;
        fy_ = fy;
        center_x_ = center_x;
        center_y_ = center_y;
    };

    void mallocDeviceMemory();
    void mallocHostMemory();
    void checkInitializedSafely();
    float2 *getWarpedPixelPosPtr() { return h_warped_pixel_pos_; };
    float *getILOldPtr(){return h_IL_old_;};
    float *getILNewPtr(){return h_IL_new_;};
    void resetToZeroILOldNew();

    void updateEventRotatedRayArr(int idx, float x, float y, float z);

    void warpEventsWrapper();
    void accumulatePolarityWrapper(float2 *h_new_warped_pixel_pos, float2 *h_old_warped_pixel_pos, int num_new_events, int num_old_events);

private:
    int subset_size_ = 0;
    float fx_ = 0, fy_ = 0, center_x_ = 0, center_y_ = 0;

    float2 *d_warped_pixels_ = nullptr;
    float3 *d_rotated_ray_ = nullptr;

    float3 *h_e_ray_cam_ = nullptr;
    float3 *h_e_ray_rotated_ = nullptr;
    float2 *h_warped_pixel_pos_ = nullptr;
    bool *h_oldevent_ = nullptr;
 
    // Panorama
    int IL_old_rows_ = 0, IL_old_cols_ = 0,  IL_new_rows_ = 0, IL_new_cols_ = 0;
    float *d_IL_old_ = nullptr;
    float *d_IL_new_ = nullptr;
    float *h_IL_old_ = nullptr;
    float *h_IL_new_ = nullptr;

    // To accumulate on pano
    // float2 *d_new_xx_yy_ = nullptr;
    // float2 *d_new_dx_dy_ = nullptr;
    // int2 *d_new_x_y_ = nullptr;

    // float2 *d_old_xx_yy_ = nullptr;
    // float2 *d_old_dx_dy_ = nullptr;
    // int2 *d_old_x_y_ = nullptr;
};

__global__ void add_kernel(float *A, float *B, float *C, int N);
bool test_cuda(int N, float *h_A, float *h_B, float *h_C);

__global__ void matrixVectorMultiply(const float *R, const float *vec, float *result);
void matrixVectorMultWrapper(const float *h_rot_m, const float *h_vec, float *h_res);

__global__ void warpEvents(float3 *h_rotated_ray, float2 *warped_pixel_pose, float2 center, float fx, float fy, int n);
__global__ void simpleAccumulateIL(float *d_IL, int IL_old_row, int IL_old_cols, int2 *d_xy, float2 *d_dxdy, int n);
__global__ void getIntAndDecimal(float2 *xx_yy, float2 *dx_dy, int2 *x_y, int n);
__global__ void setILToZero(float *d_IL, int IL_old_row, int IL_old_cols);

