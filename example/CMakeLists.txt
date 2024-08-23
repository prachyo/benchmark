option(ENABLE_OPENMP "Enable OpenMP support" OFF)
option(ENABLE_TBB "Enable TBB support" OFF)
option(ENABLE_CUDA "Enable CUDA support" OFF)
option(ENABLE_OPENBLAS "Enable OpenBLAS support" OFF)

set(EXAMPLES_DIR ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "EXAMPLES_DIR: ${EXAMPLES_DIR}")

include_directories(${sw_benchmark_INCLUDE_DIRS})

project(examples)

# Add executable for BatchGEMV.cpp
add_executable(batchgemv ${EXAMPLES_DIR}/BatchGEMV.cpp)
target_link_libraries(batchgemv swbench::swbench)

# Define macros based on the options
if(ENABLE_OPENMP)
    target_compile_definitions(batchgemv PRIVATE OPENMP_ENABLED)
    find_package(OpenMP REQUIRED)
    target_link_libraries(batchgemv OpenMP::OpenMP_CXX)
endif()

if(ENABLE_TBB)
    target_compile_definitions(batchgemv PRIVATE TBB_ENABLED)
    find_package(TBB REQUIRED)
    target_link_libraries(batchgemv TBB::tbb)
endif()

if(ENABLE_CUDA)
    target_compile_definitions(batchgemv PRIVATE CUDA_ENABLED)
    find_package(CUDA REQUIRED)
    target_link_libraries(batchgemv CUDA::cudart)
endif()

if(ENABLE_OPENBLAS)
    target_compile_definitions(batchgemv PRIVATE OPENBLAS_ENABLED)
    find_package(OpenBLAS REQUIRED)
    target_link_libraries(batchgemv OpenBLAS::OpenBLAS)
endif()