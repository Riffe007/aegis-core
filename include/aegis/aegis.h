/*
 * AEGIS Core C ABI. Pure C (C99 and later), no C++ or CUDA types.
 *
 * All functions return an int status: AEGIS_OK (0) or a negative AEGIS_ERR_*.
 * No function exits or aborts the process. aegis_status_string() describes a
 * status code; aegis_last_error_string() gives the detail of the last failure
 * on the calling thread (errno-style: kept until the next failure), and
 * aegis_last_error(ctx) the last CUDA error a context saw.
 *
 * Memory: GEMM operands are device pointers to FP16 data (IEEE binary16 bit
 * patterns, uint16_t on the host). Allocate them with aegis_device_alloc() or
 * any CUDA allocator in the same process.
 *
 * Threading: a context may be used from one thread at a time.
 *
 * Example:
 *   aegis_context* ctx;
 *   if (aegis_create(&ctx, 0) != AEGIS_OK) ...;
 *   aegis_gemm_f16(ctx, dA, dB, dC, M, N, K);
 *   aegis_synchronize(ctx);
 *   aegis_destroy(ctx);
 */
#ifndef AEGIS_AEGIS_H
#define AEGIS_AEGIS_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define AEGIS_API __declspec(dllexport)
#else
#define AEGIS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Status codes. */
#define AEGIS_OK 0
#define AEGIS_ERR_INVALID_ARGUMENT (-1)  /* null pointer, non-positive dimension */
#define AEGIS_ERR_CUDA (-2)              /* a CUDA runtime call failed */
#define AEGIS_ERR_BUFFER_TOO_SMALL (-3)  /* see the `needed` out-parameter */
#define AEGIS_ERR_NO_DEVICE (-4)         /* device index out of range / no GPU */
#define AEGIS_ERR_OUT_OF_MEMORY (-5)
#define AEGIS_ERR_LAUNCH_FAILED (-6)     /* a kernel could not be launched (since ABI 1.1) */

/* ABI version: bump the major on any incompatible change. */
#define AEGIS_ABI_VERSION_MAJOR 1
#define AEGIS_ABI_VERSION_MINOR 3
AEGIS_API int aegis_abi_version(void); /* major * 1000 + minor */

/* Static description of a status code. Never NULL. */
AEGIS_API const char* aegis_status_string(int status);

/* Detail of the last failed call on this thread, e.g. which kernel failed to
 * launch and the CUDA error text. "" if nothing has failed. Valid until the
 * next aegis_* call on this thread. Since ABI 1.1. */
AEGIS_API const char* aegis_last_error_string(void);

/* Opaque context: one device and one stream (a private non-blocking stream,
 * or the caller's, see aegis_set_stream). */
typedef struct aegis_context aegis_context;

AEGIS_API int aegis_create(aegis_context** out, int device);
AEGIS_API int aegis_destroy(aegis_context* ctx); /* NULL is a no-op */

/* Use a caller-owned cudaStream_t (passed as void*); NULL restores the
 * context's own stream. The context never destroys a caller's stream. */
AEGIS_API int aegis_set_stream(aegis_context* ctx, void* cuda_stream);

/* Wait for all work queued on the context's stream. */
AEGIS_API int aegis_synchronize(aegis_context* ctx);

/* Text of the last CUDA error seen by this context ("" if none). */
AEGIS_API const char* aegis_last_error(const aegis_context* ctx);

/* Device memory helpers (synchronous). */
AEGIS_API int aegis_device_alloc(aegis_context* ctx, size_t bytes, void** out);
AEGIS_API int aegis_device_free(aegis_context* ctx, void* ptr); /* NULL is a no-op */
AEGIS_API int aegis_copy_to_device(aegis_context* ctx, void* dst, const void* src, size_t bytes);
AEGIS_API int aegis_copy_to_host(aegis_context* ctx, void* dst, const void* src, size_t bytes);

/* C[M,N] = A[M,K] * B[K,N]; row-major, FP16 in, FP32 accumulate, FP16 out.
 * The packed row-major NN case of aegis_gemm_f16_ex (alpha 1, beta 0); since
 * ABI 1.2, M, N or K == 0 is a no-op returning AEGIS_OK (was an error).
 * Runs the tuned winner for the device (aegis::matmul), else cuBLAS.
 * Asynchronous on the context's stream. A kernel that cannot be launched for
 * this shape (e.g. its grid exceeds device limits) returns
 * AEGIS_ERR_LAUNCH_FAILED; the context stays usable. */
AEGIS_API int aegis_gemm_f16(aegis_context* ctx, const void* A, const void* B, void* C,
                             int M, int N, int K);

/* ---- General GEMM (since ABI 1.2) -------------------------------------
 * For each batch b < batch:
 *   C_b = alpha * A_b * B_b + beta * C_b
 *   A_b = A + b * stride_a: M x K, layout_a, leading dimension lda
 *   B_b = B + b * stride_b: K x N, layout_b, leading dimension ldb
 *   C_b = C + b * stride_c: M x N, row-major, leading dimension ldc (>= N)
 * Layouts: AEGIS_ROW_MAJOR (X[r*ld + c], ld >= cols) or AEGIS_COL_MAJOR
 * (X[r + c*ld], ld >= rows). Strides and leading dimensions are in elements.
 * FP16 in, FP32 accumulate, FP16 out; NaN and Inf propagate, except that
 * alpha == 0 makes A*B not contribute (C = beta*C), as in BLAS. beta == 0: C is
 * not read. Any alignment is accepted (misaligned operands take a slower path).
 * M, N, K or batch == 0: no-op, AEGIS_OK, nothing launched, C untouched.
 * Negative sizes, too-small leading dimensions, null pointers with nonzero
 * sizes, overlapping C batches: AEGIS_ERR_INVALID_ARGUMENT.
 * Asynchronous on the context's stream; runs the tuned winner for (M, N, K)
 * (tuned on row-major NN) if one exists, else cuBLAS. */
#define AEGIS_ROW_MAJOR 0
#define AEGIS_COL_MAJOR 1

AEGIS_API int aegis_gemm_f16_ex(aegis_context* ctx, int layout_a, int layout_b, int M, int N, int K,
                                float alpha, const void* A, int64_t lda, const void* B, int64_t ldb,
                                float beta, void* C, int64_t ldc);

AEGIS_API int aegis_gemm_f16_strided_batched(aegis_context* ctx, int layout_a, int layout_b, int M,
                                             int N, int K, float alpha, const void* A, int64_t lda,
                                             int64_t stride_a, const void* B, int64_t ldb,
                                             int64_t stride_b, float beta, void* C, int64_t ldc,
                                             int64_t stride_c, int batch);

/* Name of the implementation aegis_gemm_f16 would run for this shape, copied
 * NUL-terminated into buf. `needed` (optional) receives the size required,
 * including the NUL; AEGIS_ERR_BUFFER_TOO_SMALL if buf_len is less. */
AEGIS_API int aegis_gemm_choice(aegis_context* ctx, int M, int N, int K, char* buf,
                                size_t buf_len, size_t* needed);

/* JSON device manifest (the same document tools/aegis_manifest prints), copied
 * NUL-terminated into buf. The first call per device measures bandwidth
 * (~0.1 s, briefly allocates 1 GiB of device memory); later calls in the same
 * process return the same document, so a size query followed by a fill is
 * consistent. `needed` (optional) receives the size required including the
 * NUL; pass buf = NULL, buf_len = 0 to query it. */
AEGIS_API int aegis_device_manifest(int device, char* buf, size_t buf_len, size_t* needed);

/* ---- Hardware discovery (since ABI 1.3) ----------------------------------
 * What Core runs on: GPU (CUDA device properties; NVML where available, Tegra
 * sysfs otherwise), CPU (clusters and core types from /sys/devices/system/cpu,
 * cpufreq ranges, ISA features from getauxval(AT_HWCAP / AT_HWCAP2)), memory,
 * thermal zones and video engines (NVDEC / NVENC presence and codecs). Fields
 * no source could read are unknown: -1, "" or JSON null, never a guess. The
 * C++ API with every field is include/aegis/hw.h.
 */

/* The full discovery as a JSON document (schema "aegis-hw/1"; the document
 * tools/aegis_hw and `aegis-perf hw` print), copied NUL-terminated into buf.
 * Thermal readings change between calls, so a size query (buf = NULL,
 * buf_len = 0) takes a snapshot for the calling thread and the next call with
 * a buffer on that thread returns that same snapshot. `needed` (optional)
 * receives the size required including the NUL. AEGIS_ERR_NO_DEVICE if
 * `device` is not a CUDA device (the CPU, memory and thermal parts need none:
 * pass device = -1 for them alone: no CUDA call is made, and the gpu object comes
 * from Tegra sysfs or has present = false). */
AEGIS_API int aegis_hw_info_json(int device, char* buf, size_t buf_len, size_t* needed);

/* The GPU part as a struct. Set struct_size = sizeof(aegis_hw_gpu_info) before
 * the call; a later ABI may append fields, and a library fills at most
 * struct_size bytes. */
typedef struct aegis_hw_gpu_info {
  uint32_t struct_size;
  int32_t present;               /* 1: a CUDA device answered */
  int32_t device;
  int32_t arch;                  /* major * 10 + minor */
  int32_t sm_count;
  int32_t regs_per_sm;
  int64_t smem_per_sm;           /* bytes */
  int64_t smem_per_block;        /* without opt-in */
  int64_t smem_per_block_optin;
  int64_t l2_bytes;
  int64_t memory_bytes;
  int32_t unified;               /* 1: device memory is host memory (integrated) */
  int32_t tma_hardware;          /* 1: arch >= 90 */
  int32_t tma_built;             /* 1: this library compiled the TMA tier */
  int32_t nvml;                  /* 1: NVML answered for this device */
  int32_t max_sm_clock_mhz;      /* -1: unknown */
  int32_t max_mem_clock_mhz;     /* -1: unknown */
  double power_limit_w;          /* < 0: not readable */
  char name[128];
} aegis_hw_gpu_info;
AEGIS_API int aegis_hw_gpu(int device, aegis_hw_gpu_info* out);

/* The CPU part as a struct (struct_size as above). Performance cores: the core
 * type with the highest maximum frequency (then cpu_capacity). */
typedef struct aegis_hw_cpu_info {
  uint32_t struct_size;
  int32_t logical_cpus;
  int32_t clusters;
  int32_t core_types;
  int32_t performance_cores;     /* cores of the performance type(s) */
  int64_t performance_max_khz;   /* -1: unknown */
  int32_t neon, fp16, dotprod, sve, sve2, i8mm, bf16;  /* 1: present */
} aegis_hw_cpu_info;
AEGIS_API int aegis_hw_cpu(aegis_hw_cpu_info* out);

#ifdef __cplusplus
}
#endif

#endif /* AEGIS_AEGIS_H */
