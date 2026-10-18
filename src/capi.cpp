// C ABI (include/aegis/aegis.h) over aegis::matmul and the device manifest.
//
// Every entry point validates its arguments and turns CUDA, cuBLAS and launch
// errors into status codes. Nothing on this path exits the process.
#include "aegis/aegis.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <string>

#include "aegis/hw.h"
#include "aegis/manifest.h"
#include "aegis/matmul.h"

struct aegis_context {
  int device = 0;
  cudaStream_t own = nullptr;     // created by aegis_create
  cudaStream_t stream = nullptr;  // own, or the caller's
  std::string last_error;
};

namespace {

thread_local std::string thread_error;  // aegis_last_error_string()

int fail(int status, const std::string& detail) {
  thread_error = detail;
  return status;
}

int fail_cuda(aegis_context* ctx, cudaError_t e) {
  if (ctx) ctx->last_error = cudaGetErrorString(e);
  cudaGetLastError();  // clear a non-sticky error so it does not leak into the next call
  return fail(e == cudaErrorMemoryAllocation ? AEGIS_ERR_OUT_OF_MEMORY : AEGIS_ERR_CUDA,
              cudaGetErrorString(e));
}

// Make ctx's device current on this thread before any CUDA call.
int enter(aegis_context* ctx) {
  if (!ctx) return fail(AEGIS_ERR_INVALID_ARGUMENT, aegis_status_string(AEGIS_ERR_INVALID_ARGUMENT));
  const cudaError_t e = cudaSetDevice(ctx->device);
  return e == cudaSuccess ? AEGIS_OK : fail_cuda(ctx, e);
}

int copy_out(const std::string& s, char* buf, size_t buf_len, size_t* needed) {
  if (needed) *needed = s.size() + 1;
  if (!buf && buf_len == 0) return AEGIS_OK;  // size query
  if (!buf) return fail(AEGIS_ERR_INVALID_ARGUMENT, aegis_status_string(AEGIS_ERR_INVALID_ARGUMENT));
  if (buf_len < s.size() + 1) return fail(AEGIS_ERR_BUFFER_TOO_SMALL, aegis_status_string(AEGIS_ERR_BUFFER_TOO_SMALL));
  std::memcpy(buf, s.c_str(), s.size() + 1);
  return AEGIS_OK;
}

}  // namespace

extern "C" {

int aegis_abi_version(void) { return AEGIS_ABI_VERSION_MAJOR * 1000 + AEGIS_ABI_VERSION_MINOR; }

const char* aegis_last_error_string(void) { return thread_error.c_str(); }

const char* aegis_status_string(int status) {
  switch (status) {
    case AEGIS_OK: return "ok";
    case AEGIS_ERR_INVALID_ARGUMENT: return "invalid argument";
    case AEGIS_ERR_CUDA: return "CUDA error (see aegis_last_error)";
    case AEGIS_ERR_BUFFER_TOO_SMALL: return "buffer too small (see needed)";
    case AEGIS_ERR_NO_DEVICE: return "no such CUDA device";
    case AEGIS_ERR_OUT_OF_MEMORY: return "out of device memory";
    case AEGIS_ERR_LAUNCH_FAILED: return "kernel launch failed (see aegis_last_error_string)";
    default: return "unknown status";
  }
}

int aegis_create(aegis_context** out, int device) {
  if (!out) return fail(AEGIS_ERR_INVALID_ARGUMENT, aegis_status_string(AEGIS_ERR_INVALID_ARGUMENT));
  *out = nullptr;
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || device < 0 || device >= count) {
    cudaGetLastError();
    return fail(AEGIS_ERR_NO_DEVICE, aegis_status_string(AEGIS_ERR_NO_DEVICE));
  }
  auto* ctx = new (std::nothrow) aegis_context;
  if (!ctx) return fail(AEGIS_ERR_OUT_OF_MEMORY, aegis_status_string(AEGIS_ERR_OUT_OF_MEMORY));
  ctx->device = device;
  cudaError_t e = cudaSetDevice(device);
  if (e == cudaSuccess) e = cudaStreamCreateWithFlags(&ctx->own, cudaStreamNonBlocking);
  if (e != cudaSuccess) {
    const int status = fail_cuda(nullptr, e);
    delete ctx;
    return status;
  }
  ctx->stream = ctx->own;
  *out = ctx;
  return AEGIS_OK;
}

int aegis_destroy(aegis_context* ctx) {
  if (!ctx) return AEGIS_OK;
  int status = AEGIS_OK;
  if (cudaSetDevice(ctx->device) == cudaSuccess && ctx->own) {
    if (cudaStreamSynchronize(ctx->own) != cudaSuccess ||
        cudaStreamDestroy(ctx->own) != cudaSuccess) {
      status = AEGIS_ERR_CUDA;
      cudaGetLastError();
    }
  }
  delete ctx;
  return status;
}

int aegis_set_stream(aegis_context* ctx, void* cuda_stream) {
  if (!ctx) return fail(AEGIS_ERR_INVALID_ARGUMENT, aegis_status_string(AEGIS_ERR_INVALID_ARGUMENT));
  ctx->stream = cuda_stream ? static_cast<cudaStream_t>(cuda_stream) : ctx->own;
  return AEGIS_OK;
}

int aegis_synchronize(aegis_context* ctx) {
  if (int s = enter(ctx)) return s;
  const cudaError_t e = cudaStreamSynchronize(ctx->stream);
  return e == cudaSuccess ? AEGIS_OK : fail_cuda(ctx, e);
}

const char* aegis_last_error(const aegis_context* ctx) {
  return ctx ? ctx->last_error.c_str() : "";
}

int aegis_device_alloc(aegis_context* ctx, size_t bytes, void** out) {
  if (!out || bytes == 0) return fail(AEGIS_ERR_INVALID_ARGUMENT, aegis_status_string(AEGIS_ERR_INVALID_ARGUMENT));
  *out = nullptr;
  if (int s = enter(ctx)) return s;
  const cudaError_t e = cudaMalloc(out, bytes);
  return e == cudaSuccess ? AEGIS_OK : fail_cuda(ctx, e);
}

int aegis_device_free(aegis_context* ctx, void* ptr) {
  if (!ptr) return ctx ? AEGIS_OK : AEGIS_ERR_INVALID_ARGUMENT;
  if (int s = enter(ctx)) return s;
  const cudaError_t e = cudaFree(ptr);
  return e == cudaSuccess ? AEGIS_OK : fail_cuda(ctx, e);
}

static int copy(aegis_context* ctx, void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
  if (!dst || !src) return fail(AEGIS_ERR_INVALID_ARGUMENT, aegis_status_string(AEGIS_ERR_INVALID_ARGUMENT));
  if (int s = enter(ctx)) return s;
  // On the context's stream, then wait: ordered after queued GEMMs.
  cudaError_t e = cudaMemcpyAsync(dst, src, bytes, kind, ctx->stream);
  if (e == cudaSuccess) e = cudaStreamSynchronize(ctx->stream);
  return e == cudaSuccess ? AEGIS_OK : fail_cuda(ctx, e);
}

int aegis_copy_to_device(aegis_context* ctx, void* dst, const void* src, size_t bytes) {
  return copy(ctx, dst, src, bytes, cudaMemcpyHostToDevice);
}

int aegis_copy_to_host(aegis_context* ctx, void* dst, const void* src, size_t bytes) {
  return copy(ctx, dst, src, bytes, cudaMemcpyDeviceToHost);
}

namespace {

// Every GEMM entry point ends here: validate, run aegis::matmul, map errors.
int run_gemm(aegis_context* ctx, const aegis::gemm::GemmParams& p, const char* fn) {
  if (int s = enter(ctx)) return s;
  if (aegis::gemm::validate(p) != cudaSuccess)
    return fail(AEGIS_ERR_INVALID_ARGUMENT,
                std::string(fn) + ": negative size, too-small leading dimension, null operand, "
                                  "bad layout or overlapping C batches");
  if (aegis::gemm::is_empty(p)) return AEGIS_OK;
  const cudaError_t e = aegis::matmul(p, ctx->stream);
  if (e == cudaSuccess) return AEGIS_OK;
  ctx->last_error = cudaGetErrorString(e);
  const std::string detail = aegis::matmul_last_error();
  if (e == cudaErrorMemoryAllocation) return fail(AEGIS_ERR_OUT_OF_MEMORY, detail);
  if (e == cudaErrorUnknown) return fail(AEGIS_ERR_CUDA, detail);  // cuBLAS failure
  return fail(AEGIS_ERR_LAUNCH_FAILED, detail);
}

bool to_layout(int l, aegis::gemm::Layout* out) {
  if (l == AEGIS_ROW_MAJOR) *out = aegis::gemm::Layout::RowMajor;
  else if (l == AEGIS_COL_MAJOR) *out = aegis::gemm::Layout::ColMajor;
  else return false;
  return true;
}

}  // namespace

int aegis_gemm_f16(aegis_context* ctx, const void* A, const void* B, void* C, int M, int N,
                   int K) {
  return run_gemm(ctx,
                  aegis::gemm::row_major_params(static_cast<const __half*>(A), static_cast<const __half*>(B),
                                                static_cast<__half*>(C), M, N, K),
                  "aegis_gemm_f16");
}

int aegis_gemm_f16_strided_batched(aegis_context* ctx, int layout_a, int layout_b, int M, int N, int K,
                                   float alpha, const void* A, int64_t lda, int64_t stride_a, const void* B,
                                   int64_t ldb, int64_t stride_b, float beta, void* C, int64_t ldc,
                                   int64_t stride_c, int batch) {
  aegis::gemm::GemmParams p;
  if (!to_layout(layout_a, &p.layout_a) || !to_layout(layout_b, &p.layout_b))
    return fail(AEGIS_ERR_INVALID_ARGUMENT, "aegis_gemm_f16_strided_batched: layout must be "
                                            "AEGIS_ROW_MAJOR or AEGIS_COL_MAJOR");
  p.A = static_cast<const __half*>(A);
  p.B = static_cast<const __half*>(B);
  p.C = static_cast<__half*>(C);
  p.M = M; p.N = N; p.K = K;
  p.lda = lda; p.ldb = ldb; p.ldc = ldc;
  p.alpha = alpha; p.beta = beta;
  p.batch = batch;
  p.stride_a = stride_a; p.stride_b = stride_b; p.stride_c = stride_c;
  return run_gemm(ctx, p, "aegis_gemm_f16_strided_batched");
}

int aegis_gemm_f16_ex(aegis_context* ctx, int layout_a, int layout_b, int M, int N, int K, float alpha,
                      const void* A, int64_t lda, const void* B, int64_t ldb, float beta, void* C,
                      int64_t ldc) {
  return aegis_gemm_f16_strided_batched(ctx, layout_a, layout_b, M, N, K, alpha, A, lda, 0, B, ldb, 0,
                                        beta, C, ldc, 0, 1);
}

int aegis_gemm_choice(aegis_context* ctx, int M, int N, int K, char* buf, size_t buf_len,
                      size_t* needed) {
  if (M <= 0 || N <= 0 || K <= 0) return fail(AEGIS_ERR_INVALID_ARGUMENT, aegis_status_string(AEGIS_ERR_INVALID_ARGUMENT));
  if (int s = enter(ctx)) return s;
  return copy_out(aegis::matmul_choice(M, N, K), buf, buf_len, needed);
}

namespace {

// Discovery for the C ABI; device -1: no CUDA call (CPU, memory, thermal; Tegra GPU from sysfs).
int hw_discover(int device, bool video, aegis::hw::HwInfo* out) {
  aegis::hw::DiscoverOptions opt;
  opt.video = video;
  if (device < 0) {
    if (device != -1) return fail(AEGIS_ERR_NO_DEVICE, aegis_status_string(AEGIS_ERR_NO_DEVICE));
    opt.cuda = false;
    opt.video = false;
  } else {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || device >= count) {
      cudaGetLastError();
      return fail(AEGIS_ERR_NO_DEVICE, aegis_status_string(AEGIS_ERR_NO_DEVICE));
    }
    opt.device = device;
  }
  *out = aegis::hw::discover(opt);
  return AEGIS_OK;
}

}  // namespace

int aegis_hw_info_json(int device, char* buf, size_t buf_len, size_t* needed) {
  // The size query's snapshot, returned by the next fill on this thread.
  thread_local bool pending = false;
  thread_local int pending_device = 0;
  thread_local std::string snapshot;
  if (!buf && buf_len != 0) return fail(AEGIS_ERR_INVALID_ARGUMENT, "buf is NULL but buf_len is not 0");
  if (!(buf && pending && pending_device == device)) {
    aegis::hw::HwInfo info;
    if (const int s = hw_discover(device, true, &info); s != AEGIS_OK) return s;
    snapshot = aegis::hw::to_json(info);
  }
  pending = buf == nullptr;
  pending_device = device;
  return copy_out(snapshot, buf, buf_len, needed);
}

int aegis_hw_gpu(int device, aegis_hw_gpu_info* out) {
  if (!out || out->struct_size < sizeof(uint32_t)) return fail(AEGIS_ERR_INVALID_ARGUMENT, "out is NULL or struct_size unset");
  if (device < 0) return fail(AEGIS_ERR_NO_DEVICE, aegis_status_string(AEGIS_ERR_NO_DEVICE));
  aegis::hw::HwInfo info;
  if (const int s = hw_discover(device, false, &info); s != AEGIS_OK) return s;
  const aegis::hw::GpuInfo& g = info.gpu;
  aegis_hw_gpu_info r;
  std::memset(&r, 0, sizeof r);
  r.struct_size = uint32_t(sizeof r);
  r.present = g.present;
  r.device = g.device;
  r.arch = g.arch;
  r.sm_count = g.sm_count;
  r.regs_per_sm = g.regs_per_sm;
  r.smem_per_sm = g.smem_per_sm;
  r.smem_per_block = g.smem_per_block;
  r.smem_per_block_optin = g.smem_per_block_optin;
  r.l2_bytes = g.l2_bytes;
  r.memory_bytes = g.memory_bytes;
  r.unified = g.unified;
  r.tma_hardware = g.tma_hardware;
  r.tma_built = g.tma_built;
  r.nvml = g.nvml;
  r.max_sm_clock_mhz = g.max_sm_clock_mhz;
  r.max_mem_clock_mhz = g.max_mem_clock_mhz;
  r.power_limit_w = g.power_limit_w;
  std::snprintf(r.name, sizeof r.name, "%s", g.name.c_str());
  const uint32_t want = out->struct_size;
  std::memcpy(out, &r, want < sizeof r ? want : sizeof r);
  out->struct_size = want < sizeof r ? want : uint32_t(sizeof r);
  return AEGIS_OK;
}

int aegis_hw_cpu(aegis_hw_cpu_info* out) {
  if (!out || out->struct_size < sizeof(uint32_t)) return fail(AEGIS_ERR_INVALID_ARGUMENT, "out is NULL or struct_size unset");
  aegis::hw::HwInfo info;
  if (const int s = hw_discover(-1, false, &info); s != AEGIS_OK) return s;
  const aegis::hw::CpuInfo& c = info.cpu;
  aegis_hw_cpu_info r;
  std::memset(&r, 0, sizeof r);
  r.struct_size = uint32_t(sizeof r);
  r.logical_cpus = c.logical_cpus;
  r.clusters = int32_t(c.clusters.size());
  r.core_types = int32_t(c.core_types.size());
  r.performance_max_khz = -1;
  for (const aegis::hw::CoreType& t : c.core_types)
    if (t.performance) {
      r.performance_cores += int32_t(t.cpus.size());
      r.performance_max_khz = t.max_khz > r.performance_max_khz ? t.max_khz : r.performance_max_khz;
    }
  r.neon = c.isa.neon; r.fp16 = c.isa.fp16; r.dotprod = c.isa.dotprod; r.sve = c.isa.sve;
  r.sve2 = c.isa.sve2; r.i8mm = c.isa.i8mm; r.bf16 = c.isa.bf16;
  const uint32_t want = out->struct_size;
  std::memcpy(out, &r, want < sizeof r ? want : sizeof r);
  out->struct_size = want < sizeof r ? want : uint32_t(sizeof r);
  return AEGIS_OK;
}

int aegis_device_manifest(int device, char* buf, size_t buf_len, size_t* needed) {
  static std::mutex mu;
  static std::map<int, std::string> cache;  // measured once per device per process
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || device < 0 || device >= count) {
    cudaGetLastError();
    return fail(AEGIS_ERR_NO_DEVICE, aegis_status_string(AEGIS_ERR_NO_DEVICE));
  }
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(device);
  if (it == cache.end()) {
    std::string json;
    const cudaError_t e = aegis::device_manifest_json(device, json);
    if (e != cudaSuccess) return fail_cuda(nullptr, e);
    it = cache.emplace(device, std::move(json)).first;
  }
  return copy_out(it->second, buf, buf_len, needed);
}

}  // extern "C"
