# AEGIS microbenchmarks: one binary per probe, JSON lines on stdout, SASS proof
# of the timed loop on stderr. Run them all with microbench/run_all.sh.
foreach(probe ptr_chase mma_latency mma_throughput ldmatrix smem_banks cp_async dram_bw)
  add_executable(mb_${probe} ${probe}.cu)
  target_link_libraries(mb_${probe} PRIVATE CUDA::cudart)
  target_include_directories(mb_${probe} PRIVATE ${CMAKE_SOURCE_DIR}/src)
endforeach()

# Host-to-completion launch latency of the decode winner: runtime, driver
# (cuLaunchKernel), CUDA Graph, aegis::matmul. Needs the library and libcuda.
add_executable(mb_launch_overhead launch_overhead.cu)
target_link_libraries(mb_launch_overhead PRIVATE aegis CUDA::cuda_driver)
target_include_directories(mb_launch_overhead PRIVATE ${CMAKE_SOURCE_DIR}/src)
