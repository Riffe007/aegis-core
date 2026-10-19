# AEGIS microbenchmarks: one binary per probe, JSON lines on stdout, SASS proof
# of the timed loop on stderr. Run them all with microbench/run_all.sh.
foreach(probe ptr_chase mma_latency mma_throughput ldmatrix smem_banks cp_async dram_bw)
  add_executable(mb_${probe} ${probe}.cu)
  target_link_libraries(mb_${probe} PRIVATE aegis_kernels CUDA::cudart)
  target_include_directories(mb_${probe} PRIVATE ${CMAKE_SOURCE_DIR}/src)
endforeach()

# Host-to-completion launch latency of the decode winner: runtime, driver
# (cuLaunchKernel), CUDA Graph, aegis::matmul. Needs the library and libcuda.
add_executable(mb_launch_overhead launch_overhead.cu)
target_link_libraries(mb_launch_overhead PRIVATE aegis CUDA::cuda_driver)
target_include_directories(mb_launch_overhead PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Time-to-throttle profile under sustained aegis::matmul load (sustained mode; minutes long,
# not in run_all.sh): microbench/run_throttle_profile.sh.
add_executable(mb_throttle_profile throttle_profile.cu)
target_link_libraries(mb_throttle_profile PRIVATE aegis Threads::Threads)
target_include_directories(mb_throttle_profile PRIVATE ${CMAKE_SOURCE_DIR}/src)
