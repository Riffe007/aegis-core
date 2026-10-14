cmake_minimum_required(VERSION 3.24)

# Build for the GPU on this machine unless told otherwise.
#   DGX Spark: 121   Jetson Orin Nano: 87
# Must precede project(): enabling CUDA sets CMAKE_CUDA_ARCHITECTURES to the
# compiler default (e.g. 75), after which a NOT DEFINED check never fires.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(aegis_core LANGUAGES C CXX CUDA)

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
  src/gemm/mma_pipe.cu
  src/gemm/registry.cu
)
target_include_directories(aegis_kernels PUBLIC include)
# PIC: the kernels also go into the shared C ABI library.
set_target_properties(aegis_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
                                               POSITION_INDEPENDENT_CODE ON)

# aegis::matmul: tuned-winner dispatch over the kernels, cuBLAS fallback.
add_library(aegis STATIC src/matmul.cu src/manifest.cpp)
set_target_properties(aegis PROPERTIES POSITION_INDEPENDENT_CODE ON)
# matmul.cu pulls winners in with __has_include, which dependency scanning
# misses when a header first appears; glob them so `aegis-perf emit-header`
# triggers a rebuild.
file(GLOB AEGIS_WINNER_HEADERS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/include/aegis/winners/*.h)
set_source_files_properties(src/matmul.cu PROPERTIES OBJECT_DEPENDS "${AEGIS_WINNER_HEADERS}")
target_link_libraries(aegis PUBLIC aegis_kernels CUDA::cudart CUDA::cublas)

# The timing protocol (src/bench/harness.h), shared by bench_gemm and aegis_tune.
add_library(aegis_bench_harness STATIC src/bench/harness.cu src/bench/reference.cu)
target_include_directories(aegis_bench_harness PUBLIC src)
target_link_libraries(aegis_bench_harness PUBLIC aegis_kernels CUDA::cudart CUDA::cublas CUDA::cublasLt
                                           PRIVATE ${CMAKE_DL_LIBS})

add_executable(bench_gemm bench/bench_gemm.cu)
target_link_libraries(bench_gemm PRIVATE aegis_bench_harness)

# In-process autotuner over the registry (rows in bench_gemm's schema).
add_executable(aegis_tune tools/aegis_tune.cu)
target_link_libraries(aegis_tune PRIVATE aegis_bench_harness)

add_executable(aegis_manifest tools/aegis_manifest.cu)
target_link_libraries(aegis_manifest PRIVATE aegis)

# C ABI (include/aegis/aegis.h): libaegis_capi.a and libaegis_capi.so. The
# shared library exports only the AEGIS_API symbols.
add_library(aegis_capi_objs OBJECT src/capi.cpp)
target_link_libraries(aegis_capi_objs PUBLIC aegis)
set_target_properties(aegis_capi_objs PROPERTIES POSITION_INDEPENDENT_CODE ON
                                                 CXX_VISIBILITY_PRESET hidden)
add_library(aegis_capi_static STATIC $<TARGET_OBJECTS:aegis_capi_objs>)
target_link_libraries(aegis_capi_static PUBLIC aegis)
add_library(aegis_capi_shared SHARED $<TARGET_OBJECTS:aegis_capi_objs>)
target_link_libraries(aegis_capi_shared PRIVATE aegis)
# Keep symbols from the static kernel/matmul libraries out of the dynamic table.
target_link_options(aegis_capi_shared PRIVATE -Wl,--exclude-libs,ALL)
set_target_properties(aegis_capi_static aegis_capi_shared PROPERTIES OUTPUT_NAME aegis_capi)
set_target_properties(aegis_capi_shared PROPERTIES CUDA_VISIBILITY_PRESET hidden
                                                   VISIBILITY_INLINES_HIDDEN ON
                                                   VERSION 1.0 SOVERSION 1)
target_include_directories(aegis_capi_static INTERFACE include)
target_include_directories(aegis_capi_shared INTERFACE include)

# The C test, compiled as strict C11, against both libraries.
foreach(kind static shared)
  add_executable(capi_test_${kind} tests/capi_test.c)
  target_link_libraries(capi_test_${kind} PRIVATE aegis_capi_${kind} m)
  set_target_properties(capi_test_${kind} PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON
                                                      C_EXTENSIONS OFF)
  target_compile_options(capi_test_${kind} PRIVATE -Wall -Wextra -Wpedantic -Werror)
endforeach()

add_subdirectory(microbench)

add_executable(test_matmul tests/test_matmul.cu)
target_link_libraries(test_matmul PRIVATE aegis)

# Correctness: every impl on shapes that are NOT multiples of any tile size.
enable_testing()
# Keep in sync with `bench_gemm --list` (the registry); one name per instantiation.
set(AEGIS_IMPLS
  naive tiled
  regblock_64x64x8_4x4 regblock_128x128x8_8x8 regblock_128x64x16_8x4
  mma_128x128x32_2x4 mma_128x64x32_2x2 mma_64x128x32_2x4 mma_64x64x32_2x2
  mma_pipe_128x128x32_2x4_s2 mma_pipe_128x128x32_2x4_s3 mma_pipe_128x128x32_2x4_s4
  mma_pipe_128x64x32_2x2_s2 mma_pipe_128x64x32_2x2_s3 mma_pipe_128x64x32_2x2_s4
  mma_pipe_64x128x32_2x4_s2 mma_pipe_64x128x32_2x4_s3 mma_pipe_64x128x32_2x4_s4
  mma_pipe_64x64x32_2x2_s2 mma_pipe_64x64x32_2x2_s3 mma_pipe_64x64x32_2x2_s4
  mma_pipe_128x256x32_2x4_s2 mma_pipe_128x256x32_2x4_s3
  mma_pipe_256x128x32_4x2_s2 mma_pipe_256x128x32_4x2_s3
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

add_test(NAME matmul_dispatch COMMAND test_matmul)
add_test(NAME capi_static COMMAND capi_test_static)
add_test(NAME capi_shared COMMAND capi_test_shared)

# sm_87 compile gate (Jetson Orin; CLAUDE.md Decisions: compile only, nothing
# runs for sm_87 until the Jetson port). Every source with kernels is compiled
# for sm_87 SASS as an object library that is never linked or run; the
# sm87_gate test fails if any sm_87 kernel spills or uses local memory
# (STACK/LOCAL > 0), uses more than 48 KB of static shared memory (only dynamic
# smem can be opted in), or if any object is missing its sm_87 code. A source
# that does not compile for sm_87 fails the build. The pre-commit hook turns
# this on.
option(AEGIS_SM87_GATE "Also compile every kernel for sm_87 (compile only) and test its resources" OFF)
if(AEGIS_SM87_GATE)
  set(AEGIS_SM87_SOURCES
    src/gemm/naive.cu src/gemm/tiled.cu src/gemm/regblock.cu src/gemm/mma.cu src/gemm/mma_pipe.cu
    src/gemm/registry.cu src/matmul.cu src/bench/harness.cu
    microbench/ptr_chase.cu microbench/mma_latency.cu microbench/mma_throughput.cu
    microbench/ldmatrix.cu microbench/smem_banks.cu microbench/cp_async.cu microbench/dram_bw.cu
    microbench/launch_overhead.cu
    tools/aegis_manifest.cu)
  add_library(aegis_sm87_gate OBJECT ${AEGIS_SM87_SOURCES})
  set_target_properties(aegis_sm87_gate PROPERTIES CUDA_ARCHITECTURES 87-real)
  target_include_directories(aegis_sm87_gate PRIVATE include src microbench)
  target_link_libraries(aegis_sm87_gate PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cuda_driver)
  target_compile_options(aegis_sm87_gate PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-warn-spills>)
  file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/sm87_gate_objects.txt
       CONTENT "$<JOIN:$<TARGET_OBJECTS:aegis_sm87_gate>,\n>\n")
  add_test(NAME sm87_gate
           COMMAND ${CMAKE_SOURCE_DIR}/tools/sm87_gate.sh ${CMAKE_BINARY_DIR}/sm87_gate_objects.txt)
endif()
