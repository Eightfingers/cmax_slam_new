#include "backend/event_pano_warper_cuda.h"
#include "helper_cuda.h"
#include <math.h>
#include <algorithm>
#include <chrono>
#include <vector>
#include <glog/logging.h>

CudaEventWarper::CudaEventWarper()
{
}

CudaEventWarper::~CudaEventWarper()
{
    cudaFree(d_warped_pixels_);
    cudaFree(d_rotated_ray_);
    cudaFree(d_IL_old_);
    cudaFree(d_IL_new_);

    free(h_e_ray_cam_);
    free(h_e_ray_rotated_);
    free(h_warped_pixel_pos_);
    free(h_oldevent_);
    free(h_IL_old_);
    free(h_IL_new_);

}

void CudaEventWarper::resetToZeroILOldNew()
{

    setILToZero<<<(IL_old_rows_ + 255) / 256, 256>>>(d_IL_old_, IL_old_rows_, IL_old_cols_);
    setILToZero<<<(IL_new_rows_ + 255) / 256, 256>>>(d_IL_new_, IL_old_rows_, IL_old_cols_);
    checkCudaErrors(cudaDeviceSynchronize());
}

void CudaEventWarper::mallocDeviceMemory()
{
    checkCudaErrors(cudaMalloc(&d_warped_pixels_, subset_size_ * sizeof(float2)));
    checkCudaErrors(cudaMalloc(&d_rotated_ray_, subset_size_ * sizeof(float3)));

    checkCudaErrors(cudaMalloc(&d_IL_old_, IL_old_rows_ * IL_old_cols_ * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_IL_new_, IL_new_rows_ * IL_new_cols_ * sizeof(float)));
}

void CudaEventWarper::mallocHostMemory()
{
    h_e_ray_cam_ = (float3 *)malloc(subset_size_ * sizeof(float3));
    h_e_ray_rotated_ = (float3 *)malloc(subset_size_ * sizeof(float3));
    h_warped_pixel_pos_ = (float2 *)malloc(subset_size_ * sizeof(float2));
    h_oldevent_ = (bool *)malloc(subset_size_ * sizeof(bool));

    h_IL_old_ =  (float*)malloc(IL_old_rows_ * IL_old_cols_ * sizeof(float));
    h_IL_new_ =  (float*)malloc(IL_new_rows_ * IL_new_cols_ * sizeof(float));
}

void CudaEventWarper::checkInitializedSafely()
{
    if (h_e_ray_cam_ == nullptr || h_e_ray_rotated_ == nullptr || h_warped_pixel_pos_ == nullptr || h_oldevent_ == nullptr)
    {
        std::cout << "!!! FAILED TO MALLOC HOST MEMORY !!!" << std::endl;
    }

    if (d_warped_pixels_ == nullptr)
    {
        std::cout << "!!! FAILED TO CUDA MALLOC DEVICE MEMORY !!!" << std::endl;
    }

    if (fx_ == 0 || fy_ == 0 || center_x_ == 0 || center_y_ == 0)
    {
        std::cout << "!!! PARAMETERS UNINTIALIZED  !!! " << std::endl;
        std::cout << "Subset_size_: " << subset_size_ << std::endl;
        std::cout << "fx_: " << fx_ << ", fy_: " << fy_ << ", center_x_: " << center_x_ << ", center_y_: " << center_y_ << std::endl;
    }
    else
    {
        std::cout << "Subset_size_: " << subset_size_ << std::endl;
        std::cout << "fx_: " << fx_ << ", fy_: " << fy_ << ", center_x_: " << center_x_ << ", center_y_: " << center_y_ << std::endl;
    }

    if (IL_old_rows_  == 0 || IL_old_cols_ == 0 || IL_new_rows_ == 0 || IL_new_cols_ == 0)
    {
        std::cout << "Panaroma Size Not yet initialized! " << std::endl;
    }

    checkCudaErrors(cudaGetLastError());
    std::cout << "Cuda event pano warper initialized safely" << std::endl;
}

void CudaEventWarper::updateEventRotatedRayArr(int idx, float x, float y, float z)
{
    h_e_ray_rotated_[idx].x = x;
    h_e_ray_rotated_[idx].y = y;
    h_e_ray_rotated_[idx].z = z;
}

void CudaEventWarper::warpEventsWrapper()
{
    checkCudaErrors(cudaMemcpy(d_rotated_ray_, h_e_ray_rotated_, subset_size_ * sizeof(float3), cudaMemcpyHostToDevice));
    float2 center = {center_x_, center_y_};
    warpEvents<<<(subset_size_ + 255) / 256, 256>>>(d_rotated_ray_, d_warped_pixels_, center, fx_, fy_, subset_size_);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_warped_pixel_pos_, d_warped_pixels_, subset_size_ * sizeof(float2), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaGetLastError());
}

void CudaEventWarper::accumulatePolarityWrapper(float2 *h_new_warped_pixel_pos, float2 *h_old_warped_pixel_pos, int num_new_events, int num_old_events)
{
    float2 *d_new_xx_yy = nullptr;
    float2 *d_new_dx_dy = nullptr;
    int2 *d_new_x_y = nullptr;

    float2 *d_old_xx_yy = nullptr;
    float2 *d_old_dx_dy = nullptr;
    int2 *d_old_x_y = nullptr;

    if (num_new_events > 0)
    {

        checkCudaErrors(cudaMalloc(&d_new_xx_yy, num_new_events * sizeof(float2)));
        checkCudaErrors(cudaMemcpy(d_new_xx_yy, h_new_warped_pixel_pos, num_new_events * sizeof(float2), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMalloc(&d_new_dx_dy, num_new_events * sizeof(float2)));
        checkCudaErrors(cudaMalloc(&d_new_x_y, num_new_events * sizeof(int2)));

        getIntAndDecimal<<<(num_new_events + 255) / 256, 256>>>(d_new_xx_yy, d_new_dx_dy, d_new_x_y, num_new_events);
        cudaDeviceSynchronize();

        simpleAccumulateIL<<<(num_new_events + 255) / 256, 256>>>(d_IL_new_, IL_new_rows_, IL_new_cols_, d_new_x_y, d_new_dx_dy, num_new_events);
        cudaDeviceSynchronize();

        checkCudaErrors(cudaMemcpy(h_IL_new_, d_IL_new_, IL_new_rows_ * IL_new_cols_ * sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(d_new_xx_yy));
        checkCudaErrors(cudaFree(d_new_dx_dy));
        checkCudaErrors(cudaFree(d_new_x_y));
    }

    if (num_old_events > 0)
    {
        checkCudaErrors(cudaMalloc(&d_old_xx_yy, num_old_events * sizeof(float2)));
        checkCudaErrors(cudaMemcpy(d_old_xx_yy, h_old_warped_pixel_pos, num_old_events * sizeof(float2), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMalloc(&d_old_dx_dy, num_old_events * sizeof(float2)));
        checkCudaErrors(cudaMalloc(&d_old_x_y, num_old_events * sizeof(int2)));

        getIntAndDecimal<<<(num_old_events + 255) / 256, 256>>>(d_old_xx_yy, d_old_dx_dy, d_old_x_y, num_old_events);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        simpleAccumulateIL<<<(num_old_events + 255) / 256, 256>>>(d_IL_old_, IL_old_rows_, IL_old_cols_, d_old_x_y, d_old_dx_dy, num_old_events);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaMemcpy(h_IL_old_, d_IL_old_, IL_old_rows_ * IL_old_cols_ * sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(d_old_xx_yy));
        checkCudaErrors(cudaFree(d_old_dx_dy));
        checkCudaErrors(cudaFree(d_old_x_y));
    }
}

__global__ void simpleAccumulateIL(float *d_IL, int IL_old_row, int IL_old_cols, int2 *d_xy, float2 *d_dxdy, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        int xx = d_xy[idx].x;
        int yy = d_xy[idx].y;
        float dx = d_dxdy[idx].x;
        float dy = d_dxdy[idx].y;

        d_IL[(yy * IL_old_row) + xx] += (1.f - dx) * (1.f - dy);
        d_IL[(yy * IL_old_row) + (xx + 1)] += dx * (1.f - dy);
        d_IL[((yy + 1) * IL_old_row) + xx] += (1.f - dx) * dy;
        d_IL[((yy + 1) * IL_old_row) + xx + 1] += dx * dy;
    }
}

__global__ void setILToZero(float *d_IL, int IL_old_row, int IL_old_cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < IL_old_row)
    {
        for (int i = 0; i < IL_old_cols; i ++)
        {
            int index = idx * IL_old_row + i;
            d_IL[index] = 0;
        }
    }
}

__global__ void add_kernel(float *A, float *B, float *C, int N)
{
    int i = threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

bool test_cuda(int N, float *h_A, float *h_B, float *h_C)
{

    // Initialize host arrays
    for (int i = 0; i < N; i++)
    {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Allocate device memory
    float *d_A;
    float *d_B;
    float *d_C;

    checkCudaErrors(cudaMalloc(&d_A, N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_B, N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_C, N * sizeof(float)));

    // Copy host arrays to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    // auto start = high_resolution_clock::now();

    // Kernel invocation with N threads
    add_kernel<<<1, N>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify the result
    for (int i = 0; i < N; i++)
    {
        if (h_C[i] != h_A[i] + h_B[i])
        {
            std::cerr << "Error at index " << i << std::endl;
            return false;
            break;
        }
    }
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return true;
}

__global__ void warpEvents(float3 *d_rotated_ray, float2 *d_warped_pixel_pose, float2 center, float fx, float fy, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        float x = d_rotated_ray[idx].x;
        float y = d_rotated_ray[idx].y;
        float z = d_rotated_ray[idx].z;

        const float phi = atan2f(x, z);
        const float theta = asinf(y / sqrtf(x * x + y * y + z * z));
        d_warped_pixel_pose[idx].x = center.x + (phi * fx);
        d_warped_pixel_pose[idx].y = center.y + (theta * fy);
    }
}

__global__ void getIntAndDecimal(float2 *xx_yy, float2 *dx_dy, int2 *x_y, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        // Integer part
        x_y[idx].x = (int)xx_yy[idx].x;
        x_y[idx].y = (int)xx_yy[idx].y;

        // Decimal part
        dx_dy[idx].x = xx_yy[idx].x - x_y[idx].x;
        dx_dy[idx].y = xx_yy[idx].y - x_y[idx].y;
    }
}

__global__ void matrixVectorMultiply(const float *R, const float *vec, float *result)
{
    int idx = threadIdx.x;

    if (idx < 3)
    {
        result[idx] = R[idx * 3 + 0] * vec[0] +
                      R[idx * 3 + 1] * vec[1] +
                      R[idx * 3 + 2] * vec[2];
    }
}

void matrixVectorMultWrapper(const float *h_rot_m, const float *h_vec, float *h_res)
{
    // Allocate memory on the device
    float *d_R;
    float *d_vec;
    float *d_result;

    cudaMalloc(&d_R, 9 * sizeof(float));
    cudaMalloc(&d_vec, 3 * sizeof(float));
    cudaMalloc(&d_result, 3 * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_R, h_rot_m, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, h_vec, 3 * sizeof(float), cudaMemcpyHostToDevice);

    matrixVectorMultiply<<<1, 3>>>(d_R, d_vec, d_result);
    cudaMemcpy(h_res, d_result, 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_R);
    cudaFree(d_vec);
    cudaFree(d_result);
}
