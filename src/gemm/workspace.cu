// Per-stream workspace cache. See workspace.h.
#include <vector>

#include "workspace.h"

namespace aegis::gemm::detail {
namespace {

struct Buffer {
  int dev;
  cudaStream_t stream;
  WorkspaceUse use;
  size_t bytes;  // the size class
  void* ptr;
};

thread_local std::vector<Buffer> buffers;

size_t size_class(size_t bytes) {
  size_t c = size_t(64) << 10;
  while (c < bytes) c <<= 1;
  return c;
}

}  // namespace

cudaError_t stream_workspace(cudaStream_t stream, size_t bytes, void** out, WorkspaceUse use) {
  *out = nullptr;
  if (bytes == 0) return cudaErrorInvalidValue;
  int dev = 0;
  if (const cudaError_t e = cudaGetDevice(&dev); e != cudaSuccess) return e;
  const size_t cls = size_class(bytes);
  for (const Buffer& b : buffers)
    if (b.dev == dev && b.stream == stream && b.use == use && b.bytes == cls) {
      *out = b.ptr;
      return cudaSuccess;
    }
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  if (const cudaError_t e = cudaStreamIsCapturing(stream, &status); e != cudaSuccess) return e;
  if (status != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
  void* p = nullptr;
  if (const cudaError_t e = cudaMalloc(&p, cls); e != cudaSuccess) return e;
  if (const cudaError_t e = cudaMemsetAsync(p, 0, cls, stream); e != cudaSuccess) {
    cudaFree(p);
    return e;
  }
  buffers.push_back({dev, stream, use, cls, p});
  *out = p;
  return cudaSuccess;
}

size_t stream_workspace_allocations() { return buffers.size(); }

}  // namespace aegis::gemm::detail
