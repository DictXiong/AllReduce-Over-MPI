cmake_minimum_required(VERSION 3.0.2)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(aoncclproj LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra -Wreorder -fPIC -fopenmp -mavx2 -fopt-info")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)
find_package(Git)
# 生成版本描述字符串类似 TAG-X-gHASH
execute_process(COMMAND ${GIT_EXECUTABLE} describe --abbrev=6 --dirty --always --tags
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE  GIT_REPO_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# 获取最新 commit 日期，YYYY-MM-DD
execute_process(COMMAND ${GIT_EXECUTABLE} log -1 --format=%cd --date=short
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE  GIT_REPO_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# 获取最新 commit Hash
execute_process(COMMAND ${GIT_EXECUTABLE} log -1 --format=%H
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE  GIT_REPO_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_definitions( -DGIT_REPO_VERSION=\"${GIT_REPO_VERSION}\")
add_definitions( -DGIT_REPO_DATE=\"${GIT_REPO_DATE}\")
add_definitions( -DGIT_REPO_HASH=\"${GIT_REPO_HASH}\")

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
message("${MPI_INCLUDE_PATH}")

cuda_add_executable(allreduce_over_nccl benchmark.cu)
target_link_libraries(allreduce_over_nccl ${MPI_CXX_LIBRARIES} ${CUDA_LIBRARIES} glog pthread nccl)