cmake_minimum_required(VERSION 3.0.2)
project(cmax_slam
LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -O3")

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc")
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 86)
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G") # enable cuda-gdb
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-lineinfo") # enable line info
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_86 -gencode  arch=compute_86,code=sm_86)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(gsl REQUIRED gsl)

find_package(OpenCV REQUIRED)

include_directories(include)
include_directories(thirdparty/basalt-headers/thirdparty/Sophus)
include_directories(thirdparty/basalt-headers/thirdparty/cereal/include)
include_directories(thirdparty/basalt-headers/include)

# CUDA
find_package(CUDA 12.6 REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")
include_directories("${CUDA_INCLUDE_DIRS}")
# CUDA source files
cuda_add_library(cuda_functions src/backend/event_pano_warper.cu)

file(GLOB SOURCES src/frontend/*.cpp
                  src/backend/*.cpp
                  src/utils/*.cpp
                  src/*.cpp)

# C++ source files
cs_add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME}
   ${catkin_LIBRARIES}
   ${OpenCV_LIBRARIES}
   ${gsl_LIBRARIES}
   cuda_functions
)

################################
# Unit Tests
#################################

include(GoogleTest)
find_package(GTest REQUIRED)
enable_testing()

add_executable(
  unit_test
  src/unit_test.cc
)

target_link_libraries(
  unit_test
  cuda_functions
  GTest::gtest_main
)

add_test(NAME CudaTest COMMAND unit_test)

cs_export()