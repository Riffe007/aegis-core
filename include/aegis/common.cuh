#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define AEGIS_CUDA_CHECK(expr)                                                  \
  do {                                                                          \
    cudaError_t aegis_err_ = (expr);                                            \
    if (aegis_err_ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error: %s\n  at %s:%d\n  in %s\n",             \
                   cudaGetErrorString(aegis_err_), __FILE__, __LINE__, #expr);  \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

namespace aegis {

constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

}  // namespace aegis
