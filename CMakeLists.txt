cmake_minimum_required(VERSION 3.24)

# Build for the GPU on this machine unless told otherwise.
#   DGX Spark: 121   Jetson Orin Nano: 87
# Must precede project(): enabling CUDA sets CMAKE_CUDA_ARCHITECTURES to the
# compiler default (e.g. 75), after which a NOT DEFINED check never fires.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(aegis_core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# -lineinfo keeps source<->SASS correlation for Nsight Compute and sass_dump.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --expt-relaxed-constexpr -Xptxas=-v")

add_library(aegis_kernels STATIC
  src/gemm/naive.cu
  src/gemm/tiled.cu
  src/gemm/regblock.cu
  src/gemm/mma.cu
  src/gemm/registry.cu
)
target_include_directories(aegis_kernels PUBLIC include)
set_target_properties(aegis_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

add_executable(bench_gemm bench/bench_gemm.cu)
target_link_libraries(bench_gemm PRIVATE aegis_kernels CUDA::cudart CUDA::cublas)

add_executable(aegis_manifest tools/aegis_manifest.cu)
target_include_directories(aegis_manifest PRIVATE include)
target_link_libraries(aegis_manifest PRIVATE CUDA::cudart)

# Correctness: every impl on shapes that are NOT multiples of any tile size.
enable_testing()
# Keep in sync with `bench_gemm --list` (the registry); one name per instantiation.
set(AEGIS_IMPLS
  naive tiled
  regblock_64x64x8_4x4 regblock_128x128x8_8x8 regblock_128x64x16_8x4
  mma_128x128x32_2x4 mma_128x64x32_2x2 mma_64x128x32_2x4 mma_64x64x32_2x2
)
foreach(impl ${AEGIS_IMPLS})
  # 127x129x65: scalar fallback. 200x136x72: 16-byte path with partial edge tiles.
  foreach(shape "127;129;65" "200;136;72" "1;896;896" "16;4864;896" "256;256;256")
    list(GET shape 0 M)
    list(GET shape 1 N)
    list(GET shape 2 K)
    add_test(NAME gemm_${impl}_${M}x${N}x${K}
             COMMAND bench_gemm --impl ${impl} --m ${M} --n ${N} --k ${K}
                                --reps 1 --warmup 0 --check)
  endforeach()
endforeach()
