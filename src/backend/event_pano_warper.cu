#include "backend/event_pano_warper_cuda.h"
#include <math.h>
#include <algorithm>
#include <chrono>
#include <vector>

inline void safecall(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::cout << "CudaError: " << cudaGetErrorString(err) << std::endl;
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
    // auto end = high_resolution_clock::now();
    // auto duration = duration_cast<microseconds>(end - start);
    // std::cout << "Microseconds: " << duration.count() << std::endl; // 57077microseconds

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify the result
    for (int i = 0; i < N; i++)
    {
        // std::cout << "index " << i << ":" << h_C[i] << std::endl;
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

void warpEventsWrapper(float3 *h_rotated_ray, float2 *h_warped_pixel_pose, float2 center, float fx, float fy, int n)
{
    float3 *d_rotated_ray;
    float2 *d_warped_pixel_pose;

    checkCudaErrors(cudaMalloc(&d_rotated_ray, n * sizeof(float3)));
    checkCudaErrors(cudaMemcpy(d_rotated_ray, h_rotated_ray, n * sizeof(float3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&d_warped_pixel_pose, n * sizeof(float2)));

    warpEvents<<<(n + 255) / 256, 256>>>(d_rotated_ray, d_warped_pixel_pose, center, fx, fy, n);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(h_warped_pixel_pose, d_warped_pixel_pose, n * sizeof(float2), cudaMemcpyDeviceToHost));

    safecall(cudaGetLastError());
    checkCudaErrors(cudaFree(d_rotated_ray));
    checkCudaErrors(cudaFree(d_warped_pixel_pose));

    checkCudaErrors(cudaDeviceSynchronize());
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

void getIntAndDecimalWrapper(float2 *h_xx_yy, int2 *h_x_y, float2 *h_dx_dy, int n)
{
    float2 *d_xx_yy;
    checkCudaErrors(cudaMalloc(&d_xx_yy, n * sizeof(float3)));
    checkCudaErrors(cudaMemcpy(d_xx_yy, h_xx_yy, n * sizeof(float2), cudaMemcpyHostToDevice));

    float2 *d_dx_dy;
    checkCudaErrors(cudaMalloc(&d_dx_dy, n * sizeof(float2)));

    int2 *d_x_y;
    checkCudaErrors(cudaMalloc(&d_x_y, n * sizeof(int2)));

    // dim3 gridSize(( (int) n / 512) + 1);
    // dim3 blockSize(512);

    getIntAndDecimal<<<(n + 255) / 256, 256>>>(d_xx_yy, d_dx_dy, d_x_y, n);
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(h_x_y, d_x_y, n * sizeof(int2), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_xx_yy, d_xx_yy, n * sizeof(float2), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_xx_yy));
    checkCudaErrors(cudaFree(d_dx_dy));
    checkCudaErrors(cudaFree(d_x_y));
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

void accumulatePolarityWrapper(float2 *h_new_xx_yy, float2 *h_old_xx_yy, float *h_IL_old, float *h_IL_new, int2 IL_old_dim, int2 IL_new_dim, int num_new_events, int num_old_events)
{
    // Calculate integers and decimal places
    float2 *d_new_xx_yy;
    float2 *d_new_dx_dy;
    int2 *d_new_x_y;

    // Prepare device memory
    float *d_IL_new;
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    std::cout << "Free memory: " << freeMem << " bytes, Total memory: " << totalMem << " bytes" << std::endl;

    if (num_new_events > 0)
    {
        std::cout << "Going to new \n";
        checkCudaErrors(cudaMalloc(&d_new_xx_yy, num_new_events * sizeof(float2)));
        checkCudaErrors(cudaMemcpy(d_new_xx_yy, h_new_xx_yy, num_new_events * sizeof(float2), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMalloc(&d_new_dx_dy, num_new_events * sizeof(float2)));
        checkCudaErrors(cudaMalloc(&d_new_x_y, num_new_events * sizeof(int2)));

        getIntAndDecimal<<<(num_new_events + 255) / 256, 256>>>(d_new_xx_yy, d_new_dx_dy, d_new_x_y, num_new_events);

        checkCudaErrors(cudaMalloc(&d_IL_new, IL_new_dim.x * IL_new_dim.y * sizeof(float)));
        checkCudaErrors(cudaMemcpy(d_IL_new, h_IL_new, IL_new_dim.x * IL_new_dim.y * sizeof(float), cudaMemcpyHostToDevice));

        simpleAccumulateIL<<<(num_new_events + 255) / 256, 256>>>(d_IL_new, IL_old_dim.x, IL_old_dim.y, d_new_x_y, d_new_dx_dy, num_new_events);
        safecall(cudaGetLastError());

        checkCudaErrors(cudaMemcpy(h_IL_new, d_IL_new, IL_new_dim.x * IL_new_dim.y * sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(d_new_xx_yy));
        checkCudaErrors(cudaFree(d_new_dx_dy));
        checkCudaErrors(cudaFree(d_new_x_y));
        checkCudaErrors(cudaFree(d_IL_new));
    }

    float2 *d_old_xx_yy;
    float2 *d_old_dx_dy;
    int2 *d_old_x_y;
    float *d_IL_old;
    if (num_old_events > 0)
    {
        // std::cout <<"Going to dark \n";

        // cudaError_t err = cudaMalloc(&d_old_xx_yy, num_old_events * sizeof(float2));
        // if (err != cudaSuccess) {
        //     std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        //     // Handle error (e.g., free other memory, reduce allocation size)
        // }

        std::cout << "heheheheheeh ahhahah eheh \n";
        checkCudaErrors(cudaMalloc(&d_old_xx_yy, num_old_events * sizeof(float2)));
        std::cout << "Going to memcpy fk \n";
        checkCudaErrors(cudaMemcpy(d_old_xx_yy, h_old_xx_yy, num_old_events * sizeof(float2), cudaMemcpyHostToDevice));

        checkCudaErrors(cudaMalloc(&d_old_dx_dy, num_old_events * sizeof(float2)));
        checkCudaErrors(cudaMalloc(&d_old_x_y, num_old_events * sizeof(int2)));

        getIntAndDecimal<<<(num_old_events + 255) / 256, 256>>>(d_old_xx_yy, d_old_dx_dy, d_old_x_y, num_old_events);

        checkCudaErrors(cudaMalloc(&d_IL_old, IL_old_dim.x * IL_old_dim.y * sizeof(float)));
        checkCudaErrors(cudaMemcpy(d_IL_old, h_IL_old, IL_old_dim.x * IL_old_dim.y * sizeof(float), cudaMemcpyHostToDevice));

        simpleAccumulateIL<<<(num_old_events + 255) / 256, 256>>>(d_IL_old, IL_old_dim.x, IL_old_dim.y, d_old_x_y, d_old_dx_dy, num_old_events);
        safecall(cudaGetLastError());

        checkCudaErrors(cudaMemcpy(h_IL_old, d_IL_old, IL_old_dim.x * IL_old_dim.y * sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(d_old_xx_yy));
        checkCudaErrors(cudaFree(d_old_dx_dy));
        checkCudaErrors(cudaFree(d_old_x_y));
        checkCudaErrors(cudaFree(d_IL_old));
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

void matrixVectorMultWrapper(const float* h_rot_m, const float* h_vec, float* h_res)
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
