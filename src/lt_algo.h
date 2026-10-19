#pragma once
// A cuBLASLt algorithm as data: every CUBLASLT_ALGO_CONFIG_* attribute that
// identifies it, so a choice made once (aegis_tune's cublaslt_best) can be
// stored (perf DB lt_algo, winners header) and rebuilt later with
// cublasLtMatmulAlgoInit + cublasLtMatmulAlgoConfigSetAttribute, then
// validated for the actual problem with cublasLtMatmulAlgoCheck.
// Attribute types as declared in cublasLt.h (CUDA 13.0): ID, SPLITK_NUM
// int32; TILE_ID, STAGES_ID, REDUCTION_SCHEME, CTA_SWIZZLING, CUSTOM_OPTION
// uint32; INNER_SHAPE_ID, CLUSTER_SHAPE_ID uint16.
// Shared by src/bench/harness.cu (records) and src/matmul.cu (rebuilds).
#include <cstdint>
#include <cstdio>
#include <string>

#include <cublasLt.h>

#include "aegis/matmul.h"

namespace aegis::lt {

using Config = aegis::detail::LtAlgoConfig;

inline Config read_config(const cublasLtMatmulAlgo_t& algo) {
  Config c;
  int32_t id = -1, splitk = 1;
  uint32_t tile = 0, stages = 0, reduction = 0, swizzle = 0, custom = 0;
  uint16_t inner = 0, cluster = 0;
  size_t w = 0;
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof id, &w);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile, &w);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof stages, &w);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof splitk, &w);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction, sizeof reduction, &w);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof swizzle, &w);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom, sizeof custom, &w);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &inner, sizeof inner, &w);
  cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cluster, sizeof cluster, &w);
  c.id = id;
  c.tile = int(tile);
  c.stages = int(stages);
  c.splitk = splitk;
  c.reduction = int(reduction);
  c.swizzle = int(swizzle);
  c.custom = int(custom);
  c.inner_shape = int(inner);
  c.cluster_shape = int(cluster);
  return c;
}

// Split-K reduction schemes aegis::matmul may run (cublasLt.h,
// cublasLtReductionScheme_t). COMPUTE_TYPE keeps the partials in FP32 in the
// workspace and reduces them in a separate step. OUTPUT_TYPE stores them as
// FP16, and INPLACE adds them up in the output buffer "and output data type":
// both round every partial to FP16 before the sum, the error that failed fuzz
// case 1360 (seed 41) on the cuBLAS path. So: one K pass (splitk 1, no
// reduction) or COMPUTE_TYPE, nothing else.
inline constexpr uint32_t kReductionMask = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
inline bool reduction_ok(const Config& c) {
  return c.splitk <= 1 || c.reduction == int(CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE);
}

// FP16 in/out, FP32 compute and scale: the only GEMM AEGIS asks cuBLASLt for.
inline cublasStatus_t rebuild(cublasLtHandle_t lt, const Config& c, cublasLtMatmulAlgo_t* algo) {
  if (c.id < 0) return CUBLAS_STATUS_INVALID_VALUE;
  cublasStatus_t s = cublasLtMatmulAlgoInit(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F,
                                            CUDA_R_16F, c.id, algo);
  if (s != CUBLAS_STATUS_SUCCESS) return s;
  const uint32_t tile = uint32_t(c.tile), stages = uint32_t(c.stages), reduction = uint32_t(c.reduction),
                 swizzle = uint32_t(c.swizzle), custom = uint32_t(c.custom);
  const int32_t splitk = c.splitk;
  const uint16_t inner = uint16_t(c.inner_shape), cluster = uint16_t(c.cluster_shape);
  struct Attr {
    cublasLtMatmulAlgoConfigAttributes_t a;
    const void* v;
    size_t n;
  };
  const Attr attrs[] = {
      {CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof tile},
      {CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof stages},
      {CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitk, sizeof splitk},
      {CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction, sizeof reduction},
      {CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof swizzle},
      {CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom, sizeof custom},
      {CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &inner, sizeof inner},
      {CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cluster, sizeof cluster},
  };
  for (const Attr& x : attrs)
    if ((s = cublasLtMatmulAlgoConfigSetAttribute(algo, x.a, x.v, x.n)) != CUBLAS_STATUS_SUCCESS) return s;
  return CUBLAS_STATUS_SUCCESS;
}

// The JSON object stored as lt_algo (perf DB) and in bench rows' "lt" field.
inline std::string config_json(const Config& c) {
  char buf[256];
  std::snprintf(buf, sizeof buf,
                "\"algo_id\":%d,\"tile_id\":%d,\"stages_id\":%d,\"splitk\":%d,\"reduction\":%d,"
                "\"swizzle\":%d,\"custom\":%d,\"inner_shape\":%d,\"cluster_shape\":%d",
                c.id, c.tile, c.stages, c.splitk, c.reduction, c.swizzle, c.custom, c.inner_shape,
                c.cluster_shape);
  return buf;
}

}  // namespace aegis::lt
