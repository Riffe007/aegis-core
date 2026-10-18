#pragma once
// Per-stream device workspace: the plan cache behind split-K partials (gemv,
// gemv_tc) and the cuBLASLt fallback's algorithm workspace (aegis::matmul).
//
// One buffer per (host thread, device, stream, use, size class); size classes are
// powers of two from 64 KiB. A buffer is allocated (cudaMalloc) and zeroed
// (cudaMemsetAsync on that stream) on the first request of its class, then
// returned as is: never resized, never freed (process lifetime, like the
// cuBLAS handles). After that first call nothing allocates and nothing
// synchronizes with the host.
//
// Contract:
//   - Make the first call for a (stream, size class) outside stream capture.
//     A first request while `stream` is capturing returns
//     cudaErrorStreamCaptureUnsupported and touches nothing, so the capture
//     stays valid.
//   - Users of one buffer are ordered by the stream. Split-K counters are
//     returned to 0 by the kernel that used them, so the buffer is zero again
//     after every completed launch.
//   - A graph captured on a stream reuses that stream's buffers: do not run it
//     concurrently with other work that uses the same stream's workspace
//     (as with cublasSetWorkspace).
#include <cuda_runtime.h>

#include <cstddef>

namespace aegis::gemm::detail {

// Who uses a buffer. Split-K counters must be zero at every launch (each launch
// returns them to 0, so any shape can reuse a counter buffer); partials and a
// cuBLASLt algorithm's workspace hold arbitrary bytes. Never shared between uses.
enum class WorkspaceUse : int { SplitKCounters = 0, SplitKPartials = 1, CublasLt = 2 };

// *out: at least `bytes` (> 0) of device memory, 256 B aligned, owned by the cache.
cudaError_t stream_workspace(cudaStream_t stream, size_t bytes, void** out, WorkspaceUse use);

// Number of buffers this host thread has allocated (tests: no allocation after warm-up).
size_t stream_workspace_allocations();

}  // namespace aegis::gemm::detail
