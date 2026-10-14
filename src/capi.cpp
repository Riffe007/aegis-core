// C ABI (include/aegis/aegis.h) over aegis::matmul and the device manifest.
//
// Every entry point validates its arguments and turns CUDA, cuBLAS and launch
// errors into status codes. Nothing on this path exits the process.
#include "aegis/aegis.h"

#include <cuda_runtime.h>

#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <string>

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

int aegis_gemm_f16(aegis_context* ctx, const void* A, const void* B, void* C, int M, int N,
                   int K) {
  if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0)
    return fail(AEGIS_ERR_INVALID_ARGUMENT, "aegis_gemm_f16: null operand or non-positive dimension");
  if (int s = enter(ctx)) return s;
  const cudaError_t e = aegis::matmul(static_cast<const __half*>(A), static_cast<const __half*>(B),
                                      static_cast<__half*>(C), M, N, K, ctx->stream);
  if (e == cudaSuccess) return AEGIS_OK;
  ctx->last_error = cudaGetErrorString(e);
  const std::string detail = aegis::matmul_last_error();
  if (e == cudaErrorMemoryAllocation) return fail(AEGIS_ERR_OUT_OF_MEMORY, detail);
  if (e == cudaErrorUnknown) return fail(AEGIS_ERR_CUDA, detail);  // cuBLAS failure
  return fail(AEGIS_ERR_LAUNCH_FAILED, detail);
}

int aegis_gemm_choice(aegis_context* ctx, int M, int N, int K, char* buf, size_t buf_len,
                      size_t* needed) {
  if (M <= 0 || N <= 0 || K <= 0) return fail(AEGIS_ERR_INVALID_ARGUMENT, aegis_status_string(AEGIS_ERR_INVALID_ARGUMENT));
  if (int s = enter(ctx)) return s;
  return copy_out(aegis::matmul_choice(M, N, K), buf, buf_len, needed);
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
