# AEGIS microbenchmarks: one binary per probe, JSON lines on stdout, SASS proof
# of the timed loop on stderr. Run them all with microbench/run_all.sh.
foreach(probe ptr_chase mma_latency mma_throughput ldmatrix smem_banks cp_async dram_bw)
  add_executable(mb_${probe} ${probe}.cu)
  target_link_libraries(mb_${probe} PRIVATE CUDA::cudart)
endforeach()
