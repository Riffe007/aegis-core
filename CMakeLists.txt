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

# TMA (tier 5, mma_tma) needs sm_90+: registered (AEGIS_TMA) only when every
# target architecture has it. sm_87 (Jetson) builds compile the family to empty
# kernels and keep tier 4 (cp.async); the sm87_gate checks they compile.
set(AEGIS_ARCHS "${CMAKE_CUDA_ARCHITECTURES}")
if(AEGIS_ARCHS STREQUAL "native")
  set(AEGIS_ARCHS "${CMAKE_CUDA_ARCHITECTURES_NATIVE}")
endif()
set(AEGIS_TMA ON)
foreach(arch ${AEGIS_ARCHS})
  string(REGEX MATCH "^[0-9]+" arch_num "${arch}")
  if(arch_num LESS 90)
    set(AEGIS_TMA OFF)
  endif()
endforeach()
message(STATUS "AEGIS: CUDA architectures ${AEGIS_ARCHS}; TMA family (mma_tma): ${AEGIS_TMA}")

add_library(aegis_kernels STATIC
  src/gemm/naive.cu
  src/gemm/tiled.cu
  src/gemm/regblock.cu
  src/gemm/mma.cu
  src/gemm/mma_pipe.cu
  src/gemm/mma_tma.cu
  src/gemm/gemv.cu
  src/gemm/gemv_tc.cu
  src/gemm/workspace.cu
  src/gemm/registry.cu
  src/hw/device.cu
)
target_include_directories(aegis_kernels PUBLIC include)
target_include_directories(aegis_kernels PRIVATE src)
if(AEGIS_TMA)
  target_compile_definitions(aegis_kernels PUBLIC AEGIS_TMA=1)
endif()
# PIC: the kernels also go into the shared C ABI library.
set_target_properties(aegis_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
                                               POSITION_INDEPENDENT_CODE ON)

# aegis::matmul: tuned-winner dispatch over the kernels; cuBLASLt below the dispatch floor, cuBLAS fallback.
add_library(aegis STATIC src/matmul.cu src/manifest.cpp)
set_target_properties(aegis PROPERTIES POSITION_INDEPENDENT_CODE ON)
# matmul.cu pulls winners in with __has_include, which dependency scanning
# misses when a header first appears; glob them so `aegis-perf emit-header`
# triggers a rebuild.
file(GLOB AEGIS_WINNER_HEADERS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/include/aegis/winners/*.h)
set_source_files_properties(src/matmul.cu PROPERTIES OBJECT_DEPENDS "${AEGIS_WINNER_HEADERS}")
target_link_libraries(aegis PUBLIC aegis_kernels CUDA::cudart CUDA::cublas CUDA::cublasLt)

# The timing protocol (src/bench/harness.h), shared by bench_gemm and aegis_tune.
add_library(aegis_bench_harness STATIC src/bench/harness.cu src/bench/reference.cu)
target_include_directories(aegis_bench_harness PUBLIC src)
target_link_libraries(aegis_bench_harness PUBLIC aegis_kernels CUDA::cudart CUDA::cublas CUDA::cublasLt
                                           PRIVATE ${CMAKE_DL_LIBS})

add_executable(bench_gemm bench/bench_gemm.cu)
target_link_libraries(bench_gemm PRIVATE aegis_bench_harness aegis)

# In-process autotuner over the registry (rows in bench_gemm's schema).
add_executable(aegis_tune tools/aegis_tune.cu)
target_link_libraries(aegis_tune PRIVATE aegis_bench_harness)

# Randomized correctness fuzzer (Production bar): 2,000 cases in ctest, 50,000 in
# the fuzz_nightly target (NaN-poisoned memory, under tools/gpu_job.sh).
add_executable(aegis_fuzz tools/aegis_fuzz.cu)
target_link_libraries(aegis_fuzz PRIVATE aegis_bench_harness aegis)
add_custom_target(fuzz_nightly COMMAND ${CMAKE_SOURCE_DIR}/tools/gpu_job.sh $<TARGET_FILE:aegis_fuzz>
                                       --cases 50000 --seed 20261015 --poison
                  DEPENDS aegis_fuzz USES_TERMINAL
                  COMMENT "aegis_fuzz: 50,000 cases")

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
target_link_libraries(test_matmul PRIVATE aegis aegis_bench_harness)

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
  mma_pipe_128x256x32_2x4_s2 mma_pipe_128x256x32_2x4_s3 mma_pipe_128x256x32_2x4_s4
  mma_pipe_256x128x32_4x2_s2 mma_pipe_256x128x32_4x2_s3 mma_pipe_256x128x32_4x2_s4
  gemv_64x64_v8_ksm gemv_64x128_v8_ksm gemv_128x128_v8_ksm gemv_128x256_v8_ksm gemv_256x256_v8_ksm gemv_64x128_v4_ksm gemv_128x256_v4_ksm 
  gemv_tc_16x64x64_1x4_s4_ksm gemv_tc_16x64x64_1x4_s5_ksm gemv_tc_16x64x64_1x4_s6_ksm gemv_tc_16x128x64_1x4_s4_ksm gemv_tc_16x64x32_1x4_s6_ksm gemv_tc_16x64x32_1x4_s8_ksm gemv_tc_16x32x64_1x2_s4_ksm gemv_tc_16x32x64_1x2_s6_ksm gemv_tc_16x128x32_1x4_s4_ksm gemv_tc_16x128x32_1x4_s6_ksm gemv_tc_16x64x64_1x2_s4_ksm gemv_tc_16x256x32_1x8_s4_ksm 
)
if(AEGIS_TMA)
  list(APPEND AEGIS_IMPLS
    mma_tma_128x128x32_2x4_s3 mma_tma_128x128x32_2x4_s4 mma_tma_128x128x32_2x4_s6
    mma_tma_128x128x64_2x4_s2 mma_tma_128x128x64_2x4_s3 mma_tma_256x128x32_4x2_s3
    mma_tma_256x128x32_4x2_s4 mma_tma_128x256x32_2x4_s4 mma_tma_256x128x64_4x2_s2)
endif()
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
add_test(NAME fuzz_2000 COMMAND aegis_fuzz --cases 2000 --seed 1)
add_test(NAME fuzz_selftest COMMAND aegis_fuzz --cases 200 --seed 3 --corrupt)
add_test(NAME fuzz_poison COMMAND aegis_fuzz --cases 2000 --seed 21 --poison)
# Seed 20261014 case 24632: cuBLASLt is 2.57 ulp off FP64 there (ours 0.43); the
# cuBLASLt-reference failure must be overruled by the FP64 reference.
add_test(NAME fuzz_lt_reference_error COMMAND aegis_fuzz --seed 20261014 --case 24632 --poison)
add_test(NAME fuzz_modes_same_cases COMMAND ${CMAKE_SOURCE_DIR}/tests/test_fuzz_modes.sh $<TARGET_FILE:aegis_fuzz>)

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
    src/gemm/naive.cu src/gemm/tiled.cu src/gemm/regblock.cu src/gemm/mma.cu src/gemm/mma_pipe.cu src/gemm/mma_tma.cu
    src/gemm/gemv.cu src/gemm/gemv_tc.cu src/gemm/workspace.cu src/gemm/registry.cu src/hw/device.cu src/matmul.cu src/bench/harness.cu
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

# compute-sanitizer memcheck on a subset (Production bar; the full suite under
# all four tools: tools/sanitize.sh).
find_program(AEGIS_COMPUTE_SANITIZER compute-sanitizer HINTS ${CUDAToolkit_BIN_DIR})
if(AEGIS_COMPUTE_SANITIZER)
  add_test(NAME memcheck_fuzz
           COMMAND ${CMAKE_SOURCE_DIR}/tools/gpu_job.sh ${AEGIS_COMPUTE_SANITIZER} --tool memcheck --error-exitcode 99
                   $<TARGET_FILE:aegis_fuzz> --cases 60 --seed 5 --max-dim 512)
  add_test(NAME memcheck_capi
           COMMAND ${CMAKE_SOURCE_DIR}/tools/gpu_job.sh ${AEGIS_COMPUTE_SANITIZER} --tool memcheck --error-exitcode 99 $<TARGET_FILE:capi_test_shared>)
  add_test(NAME memcheck_bench_odd
           COMMAND ${CMAKE_SOURCE_DIR}/tools/gpu_job.sh ${AEGIS_COMPUTE_SANITIZER} --tool memcheck --error-exitcode 99
                   $<TARGET_FILE:bench_gemm> --impl mma_pipe_128x256x32_2x4_s3 --m 200 --n 136 --k 72
                   --reps 1 --warmup 0 --check)
  add_test(NAME memcheck_bench_scalar
           COMMAND ${CMAKE_SOURCE_DIR}/tools/gpu_job.sh ${AEGIS_COMPUTE_SANITIZER} --tool memcheck --error-exitcode 99
                   $<TARGET_FILE:bench_gemm> --impl mma_pipe_64x64x32_2x2_s4 --m 127 --n 129 --k 65
                   --reps 1 --warmup 0 --check)
endif()

# The in-process GPU allocation budget (src/bench/gpu_budget.h, CLAUDE.md rule 13).
add_executable(test_gpu_budget tests/test_gpu_budget.cu)
target_include_directories(test_gpu_budget PRIVATE src)
target_link_libraries(test_gpu_budget PRIVATE CUDA::cudart)
target_include_directories(test_matmul PRIVATE src)
add_test(NAME gpu_budget_default COMMAND test_gpu_budget default)
add_test(NAME gpu_budget_budget COMMAND test_gpu_budget budget)
set_tests_properties(gpu_budget_budget PROPERTIES ENVIRONMENT "AEGIS_GPU_BUDGET=64M")
add_test(NAME gpu_budget_minfree COMMAND test_gpu_budget minfree)
set_tests_properties(gpu_budget_minfree PROPERTIES ENVIRONMENT "AEGIS_GPU_MIN_FREE=1000000G")

# The blocking idle guard: busy aborts (exit 3) unless --allow-busy; rows carry the reading.
add_test(NAME idle_guard_paths
         COMMAND ${CMAKE_SOURCE_DIR}/tests/test_idle_guard.sh $<TARGET_FILE:bench_gemm> $<TARGET_FILE:aegis_tune>)
# The idle guard during a run: the run's own processes are excluded, others
# over the limit make the run disturbed.
add_test(NAME idle_watch_paths COMMAND ${CMAKE_SOURCE_DIR}/tests/test_idle_watch.sh $<TARGET_FILE:bench_gemm>)
# It watches for other GPU users, so no other test may run beside it (ctest -j).
set_tests_properties(idle_watch_paths PROPERTIES RUN_SERIAL TRUE)
add_executable(test_idle_rule tests/test_idle_rule.cu)
target_link_libraries(test_idle_rule PRIVATE aegis_bench_harness)
add_test(NAME idle_rule COMMAND test_idle_rule)

# mma_tma: TMA path vs the cp.async fallback (misaligned pointers, ld, strides)
# and boundary tiles via TMA zero fill, every layout, against the FP64 reference.
if(AEGIS_TMA)
  add_executable(test_tma tests/test_tma.cu)
  target_link_libraries(test_tma PRIVATE aegis_bench_harness)
  add_test(NAME tma_paths COMMAND test_tma)
endif()

# Split-K (gemv, gemv_tc): bitwise determinism over 10 runs, strict-mode graph
# capture (no allocation after the first call), effective split counts.
add_executable(test_splitk tests/test_splitk.cu)
target_link_libraries(test_splitk PRIVATE aegis aegis_bench_harness)
add_test(NAME splitk_determinism COMMAND test_splitk)
# Launch sizing from the detected SM count (aegis-hw step 1): the decode family's
# S and grid under simulated 24 / 48 / 96-SM profiles, and matmul correct under each.
add_executable(test_hw_sizing tests/test_hw_sizing.cu)
target_link_libraries(test_hw_sizing PRIVATE aegis aegis_bench_harness)
target_include_directories(test_hw_sizing PRIVATE src)
add_test(NAME hw_sizing COMMAND test_hw_sizing)
# Decode fuzz: the GEMV family and aegis::matmul, M 1..40 (mostly <= 16), K up
# to 16384, every layout, boundaries, alpha/beta, poisoned memory.
add_test(NAME fuzz_decode COMMAND aegis_fuzz --cases 2000 --seed 41 --poison --decode --impls gemv_,matmul)

# GPU lock (CLAUDE.md rule 15): every test except the compile-only sm87_gate
# touches the GPU, is labelled "gpu" and requires the gpu_lock_guard fixture,
# which fails unless this ctest run holds the GPU lock (tools/gpu_lock.sh
# ctest ...). ctest -LE gpu runs the rest without the lock.
add_test(NAME gpu_lock_guard COMMAND ${CMAKE_SOURCE_DIR}/tools/gpu_lock_guard.sh)
set_tests_properties(gpu_lock_guard PROPERTIES FIXTURES_SETUP gpu_lock LABELS gpu)
get_property(AEGIS_ALL_TESTS DIRECTORY PROPERTY TESTS)
foreach(t ${AEGIS_ALL_TESTS})
  if(NOT t STREQUAL "sm87_gate" AND NOT t STREQUAL "gpu_lock_guard")
    set_tests_properties(${t} PROPERTIES LABELS gpu FIXTURES_REQUIRED gpu_lock)
  endif()
endforeach()
