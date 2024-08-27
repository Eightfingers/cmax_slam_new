#include <gtest/gtest.h>
#include "backend/event_pano_warper_cuda.h"
#include "backend/trajectory.h"
#include "backend/equirectangular_camera.h"
#include "utils/image_geom_util.h"
#include "utils/image_utils.h"
#include "utils/parameters.h"
#include <cmath>

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions)
{
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

TEST(PhonkTest, CudaPanoProjection)
{
  int N = 100;
  float tolerance = 1e-3f;

  // Run kernel w 256 threads
  // Allocate host memory
  float *h_A = (float *)malloc(N * sizeof(float));
  float *h_B = (float *)malloc(N * sizeof(float));
  float *h_C = (float *)malloc(N * sizeof(float));

  std::cout << "Test Vector Addition" << std::endl;
  // Vector Add 2 arrays

  EXPECT_EQ(test_cuda(N, h_A, h_B, h_C), true);
  // Free host memory>>
  free(h_A);
  free(h_B);
  free(h_C);
  std::cout << "Vector Addition completed" << std::endl;

  std::cout << "Test matrixVecMultiply " << std::endl;
  // Test matrix mul
  float h_R[9] = {1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0};
  float h_vec[3] = {1.0, 2.0, 3.0};
  float h_result[3];
  float h_expected[3] = {1.0, 2.0, 3.0};

  matrixVectorMultWrapper(h_R, h_vec, h_result);
  std::cout << "Test matrixVecMultiply completed" << std::endl;

  // Text matrix mul 2 
  for (int i = 0; i < 3; i++)
  {
    ASSERT_NEAR(h_result[i], h_expected[i], tolerance);
  }

  float h_R2[9] = {0.0, -1.0, 0.0,
                  1.0, 0.0, 0.0,
                  0.0, 0.0, 1.0};
  float h_vec2[3] = {1.0, 0.0, 0.0};
  float h_result2[3];
  float h_expected2[3] = {0.0, 1.0, 0.0};

  matrixVectorMultWrapper(h_R2, h_vec2, h_result2);

  for (int i = 0; i < 3; i++)
  {
    ASSERT_NEAR(h_result2[i], h_expected2[i], tolerance);
  }

  // Input Answers
  std::vector<Eigen::Vector3d> e_ray_w_sample = {
      Eigen::Vector3d(0.330601, -0.140709, 1.0),
      Eigen::Vector3d(-0.601842, -0.077737, 1.0),
      Eigen::Vector3d(-0.204978, 0.0321258, 1.0),
      Eigen::Vector3d(0.019904, 0.314331, 1.0),
      Eigen::Vector3d(0.307633, -0.129375, 1.0),
  };

  // Expected Outputs
  std::vector<cv::Point2d> px_mosaic_sample = {
      cv::Point2d(564.036, 234.355),
      cv::Point2d(423.705, 245.161),
      cv::Point2d(479.05, 261.127),
      cv::Point2d(515.243, 305.625),
      cv::Point2d(560.639, 235.949),
  };

  // yy, xx
  std::vector<float> IL_out1 = {
      0.621651,
      0.247642,
      0.828793,
      0.283656,
      0.0184395};

  // yy, xx + 1
  std::vector<float> IL_out2 = {
      0.0232337,
      0.591236,
      0.0438564,
      0.0912641,
      0.0326196};

  // yy + 1, xx
  std::vector<float> IL_out3 = {
      0.342321,
      0.0475642,
      0.12095,
      0.472921,
      0.342701};

  // yy + 1, xx + 1
  std::vector<float> IL_out4 = {
      0.012794,
      0.113558,
      0.0064002,
      0.152159,
      0.60624};

  // actual IL out indexes
  std::vector<int> x_vec = {
      234, 245, 261, 305, 235};

  std::vector<int> y_vec = {
      564,
      423,
      479,
      515,
      560};

  // Test Warping of events
  int subset_size = 5;

  float3 *h_e_ray_rotated = (float3 *)malloc(subset_size * sizeof(float3));
  float2 *h_warped_pixel_pos = (float2 *)malloc(subset_size * sizeof(float2));
  float2 *h_warped_pixel_pos2 = (float2 *)malloc(subset_size * sizeof(float2));

  for (int i = 0; i < subset_size; i++)
  {
    h_e_ray_rotated[i].x = e_ray_w_sample[i][0];
    h_e_ray_rotated[i].y = e_ray_w_sample[i][1];
    h_e_ray_rotated[i].z = e_ray_w_sample[i][2];
  }
  double fx = 162.975;
  double fy = 162.975;
  float2 center_{512, 256};

  // Warp event pixels to get new warped pixel positions
  std::cout << "Test warpEventsWrapper" << std::endl;
  warpEventsWrapper(h_e_ray_rotated, h_warped_pixel_pos, center_, fx, fy, subset_size);
  for (int i = 0; i < subset_size; i++)
  {
    // ASSERT_NEAR(value1, value2, tolerance);  // Custom tolerance for floats
    ASSERT_NEAR(h_warped_pixel_pos[i].x, px_mosaic_sample[i].x, tolerance);
    ASSERT_NEAR(h_warped_pixel_pos[i].y, px_mosaic_sample[i].y, tolerance);
  }
  std::cout << "Test warpEventsWrapper finished" << std::endl;

  memcpy(h_warped_pixel_pos2, h_warped_pixel_pos, subset_size * sizeof(float2));
  // Prepare data for bilinear fitting of events
  int IL_old_rows = 512;
  int IL_old_cols = 1024;

  int2 IL_old_dim{IL_old_rows, IL_old_cols};
  int2 IL_new_dim{IL_old_rows, IL_old_cols};

  int IL_old_num_pixels = IL_old_rows * IL_old_cols;
  float *h_IL_old = (float *)malloc(IL_old_num_pixels * sizeof(float));
  for (int i = 0; i < IL_old_rows; i++)
  {
    for (int j = 0; j < IL_old_cols; j++)
    {
      int idx = (i * IL_old_rows) + j;
      h_IL_old[idx] = 0;
    }
  }

  float *h_IL_new = (float *)malloc(IL_old_num_pixels * sizeof(float));
  for (int i = 0; i < IL_old_rows; i++)
  {
    for (int j = 0; j < IL_old_cols; j++)
    {
      int idx = (i * IL_old_rows) + j;
      h_IL_new[idx] = 0;
    }
  }

  std::cout << "Test accumulatePolarityWrapper" << std::endl;
  accumulatePolarityWrapper(h_warped_pixel_pos, h_warped_pixel_pos2, h_IL_old, h_IL_new, IL_old_dim, IL_new_dim, subset_size, subset_size);

  for (int i = 0; i < x_vec.size(); i++)
  {
    int x1 = x_vec[i];
    int y1 = y_vec[i];
    int idx1 = (x1 * IL_old_rows) + y1;
    EXPECT_NEAR(h_IL_old[idx1], IL_out1[i], tolerance);

    int x2 = x1;
    int y2 = y1 + 1;
    int idx2 = (x2 * IL_old_rows) + y2;
    EXPECT_NEAR(h_IL_old[idx2], IL_out2[i], tolerance);

    int x3 = x1 + 1;
    int y3 = y1;
    int idx3 = (x3 * IL_old_rows) + y3;
    EXPECT_NEAR(h_IL_old[idx3], IL_out3[i], tolerance);

    int x4 = x1 + 1;
    int y4 = y1 + 1;
    int idx4 = (x4 * IL_old_rows) + y4;
    ASSERT_NEAR(h_IL_old[idx4], IL_out4[i], tolerance);
  }
  std::cout << "Test accumulatePolarityWrapper finished" << std::endl;

  std::cout << "Unit tests end of line" << std::endl;
  // std::cout << h_IL_old[index] << "," << IL_out1[0] << std::endl;
  // ASSERT_NEAR(h_warped_pixel_pos[i].x, px_mosaic_sample[i].x, tolerance);
  // ASSERT_NEAR(h_warped_pixel_pos[i].x, px_mosaic_sample[i].x, tolerance);
}