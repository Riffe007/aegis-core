// Probe: shared-memory bank conflicts, measured as warp-wide load throughput.
//
// v1 measured 2 cycles per conflict-free 32-bit warp load: one XOR chain
// through every load and loop overhead made it issue/latency-bound, not
// bank-bound. v2:
//   - ILP independent loads per iteration (4 or 8), each into its own
//     accumulator, so nothing serializes the loads;
//   - an empty-loop calibration kernel (same loop, no loads) whose cycles are
//     subtracted;
//   - an LDS.128 variant (ld.volatile.shared.v4.u32).
//   - warps per SM 8 and 32.
// lds32: lane l loads word l*stride (stride 1..32 words); predicted ways =
// gcd(stride, 32); ideal conflict-free cost 1 cycle/warp load (128 B/clk).
// lds128: lane l loads 16 B at l*stride*16 B (stride 1..8 16-B units); a warp
// is served in 4 phases of 8 lanes; predicted ways per phase = gcd(stride, 8);
// ideal conflict-free cost 4 cycles/warp load.
// A uniform per-iteration offset keeps every lane on the same bank.
// Loads are ld.volatile: plain ld.shared with loop-invariant addresses got
// hoisted out of the loop by ptxas (the SASS proof caught it in v1).
//
// Output: probe "smem_bank_conflict": cycles per warp load (per SM, empty-loop
// subtracted); probe "smem_bank_conflict_fit": least-squares fit
// cycles = intercept + slope * predicted_ways per (width, ilp, warps).
#include <cstdint>
#include <numeric>

#include "common.cuh"

namespace {
constexpr int kWords = 32 * 32 + 8 * 32;        // lds32: max lane offset + unrolled offsets
constexpr int kBytes128 = 32 * 8 * 16 + 8 * 512;  // lds128
}  // namespace

template <int ILP>
__device__ __forceinline__ void lds32_body(int stride, int iters, uint32_t* sink, long long* cycles) {
  __shared__ uint32_t smem[kWords];
  for (int i = threadIdx.x; i < kWords; i += blockDim.x) smem[i] = i * 2654435761u;
  __syncthreads();
  const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(smem)) + (threadIdx.x % 32) * stride * 4;
  uint32_t acc[ILP] = {};
  __syncthreads();
  const long long t0 = mb::clock64_volatile();
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      uint32_t v;
      asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v) : "r"(base + j * 128));
      acc[j] ^= v;
    }
  }
  __syncthreads();
  const long long t1 = mb::clock64_volatile();
  uint32_t a = 0;
#pragma unroll
  for (int j = 0; j < ILP; ++j) a ^= acc[j];
  sink[blockIdx.x * blockDim.x + threadIdx.x] = a;
  if (blockIdx.x == 0 && threadIdx.x == 0) *cycles = t1 - t0;
}

template <int ILP>
__device__ __forceinline__ void lds128_body(int stride, int iters, uint32_t* sink, long long* cycles) {
  __shared__ __align__(16) uint8_t smem[kBytes128];
  for (int i = threadIdx.x; i < kBytes128; i += blockDim.x) smem[i] = uint8_t(i * 131);
  __syncthreads();
  const uint32_t base = static_cast<uint32_t>(__cvta_generic_to_shared(smem)) + (threadIdx.x % 32) * stride * 16;
  uint32_t acc[ILP] = {};
  __syncthreads();
  const long long t0 = mb::clock64_volatile();
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < ILP; ++j) {
      uint32_t x, y, z, w;
      asm volatile("ld.volatile.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(base + j * 512));
      acc[j] ^= x ^ y ^ z ^ w;
    }
  }
  __syncthreads();
  const long long t1 = mb::clock64_volatile();
  uint32_t a = 0;
#pragma unroll
  for (int j = 0; j < ILP; ++j) a ^= acc[j];
  sink[blockIdx.x * blockDim.x + threadIdx.x] = a;
  if (blockIdx.x == 0 && threadIdx.x == 0) *cycles = t1 - t0;
}

extern "C" __global__ void mb_smem_lds32_ilp4(int s, int it, uint32_t* o, long long* c) { lds32_body<4>(s, it, o, c); }
extern "C" __global__ void mb_smem_lds32_ilp8(int s, int it, uint32_t* o, long long* c) { lds32_body<8>(s, it, o, c); }
extern "C" __global__ void mb_smem_lds128_ilp4(int s, int it, uint32_t* o, long long* c) { lds128_body<4>(s, it, o, c); }

// Calibration: the same loop shape with no shared loads. The volatile asm
// keeps the loop (and its counter) alive.
extern "C" __global__ void mb_smem_empty(int s, int iters, uint32_t* sink, long long* cycles) {
  uint32_t acc = s;
  __syncthreads();
  const long long t0 = mb::clock64_volatile();
#pragma unroll 1
  for (int i = 0; i < iters; ++i) asm volatile("add.u32 %0, %0, 1;" : "+r"(acc));
  __syncthreads();
  const long long t1 = mb::clock64_volatile();
  sink[blockIdx.x * blockDim.x + threadIdx.x] = acc;
  if (blockIdx.x == 0 && threadIdx.x == 0) *cycles = t1 - t0;
}

namespace {

using Kernel = void (*)(int, int, uint32_t*, long long*);

struct Variant {
  const char* name;   // param "width"
  const char* kernel;
  Kernel fn;
  int ilp, max_stride, ways_mod;  // predicted ways = gcd(stride, ways_mod)
};

struct Fit { double a, b, r2; };

Fit fit(const std::vector<double>& x, const std::vector<double>& y) {
  const double n = x.size();
  const double mx = std::accumulate(x.begin(), x.end(), 0.0) / n;
  const double my = std::accumulate(y.begin(), y.end(), 0.0) / n;
  double sxy = 0, sxx = 0, syy = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) * (x[i] - mx);
    syy += (y[i] - my) * (y[i] - my);
  }
  const double b = sxx > 0 ? sxy / sxx : 0, a = my - b * mx;
  return {a, b, syy > 0 ? (sxy * sxy) / (sxx * syy) : 1.0};
}

}  // namespace

int main() {
  const mb::Device dev = mb::device();
  const Variant variants[] = {
      {"lds32", "mb_smem_lds32_ilp4", mb_smem_lds32_ilp4, 4, 32, 32},
      {"lds32", "mb_smem_lds32_ilp8", mb_smem_lds32_ilp8, 8, 32, 32},
      {"lds128", "mb_smem_lds128_ilp4", mb_smem_lds128_ilp4, 4, 8, 8},
  };
  for (const Variant& v : variants) mb::prove(v.kernel, {{"LDS", v.ilp}});
  mb::prove("mb_smem_empty", {{"LDS", 0}});

  const int sms = dev.prop.multiProcessorCount;
  uint32_t* sink;
  long long* dcycles;
  MB_CHECK(cudaMalloc(&sink, size_t(sms) * 1024 * sizeof(uint32_t)));
  MB_CHECK(cudaMalloc(&dcycles, sizeof(long long)));
  const int iters = 1 << 14;

  auto run = [&](Kernel k, int stride, int warps) {
    k<<<sms, warps * 32>>>(stride, 16, sink, dcycles);  // warm-up
    k<<<sms, warps * 32>>>(stride, iters, sink, dcycles);
    MB_CHECK(cudaGetLastError());
    long long c = 0;
    MB_CHECK(cudaMemcpy(&c, dcycles, sizeof c, cudaMemcpyDeviceToHost));
    return double(c);
  };

  for (int warps : {8, 32}) {
    const double empty = run(mb_smem_empty, 1, warps);
    mb::emit(dev, "smem_bank_conflict_calibration", "{\"warps_per_sm\":" + std::to_string(warps) + "}",
             empty / iters, "cycles/iter");
    for (const Variant& v : variants) {
      std::vector<double> ways, cost;
      for (int stride = 1; stride <= v.max_stride; ++stride) {
        const double warp_loads = double(warps) * iters * v.ilp;
        const double per = (run(v.fn, stride, warps) - empty) / warp_loads;
        const int w = std::gcd(stride, v.ways_mod);
        ways.push_back(w);
        cost.push_back(per);
        mb::emit(dev, "smem_bank_conflict",
                 std::string("{\"width\":\"") + v.name + "\",\"ilp\":" + std::to_string(v.ilp) +
                     ",\"warps_per_sm\":" + std::to_string(warps) + ",\"stride\":" + std::to_string(stride) +
                     ",\"predicted_ways\":" + std::to_string(w) + ",\"empty_loop_subtracted\":true}",
                 per, "cycles/warp_load");
      }
      const Fit f = fit(ways, cost);
      const std::string head = std::string("{\"width\":\"") + v.name + "\",\"ilp\":" + std::to_string(v.ilp) +
                               ",\"warps_per_sm\":" + std::to_string(warps) + ",\"coef\":";
      mb::emit(dev, "smem_bank_conflict_fit", head + "\"intercept\"}", f.a, "cycles/warp_load");
      mb::emit(dev, "smem_bank_conflict_fit", head + "\"slope\"}", f.b, "cycles/way");
      mb::emit(dev, "smem_bank_conflict_fit", head + "\"r2\"}", f.r2, "1");
    }
  }
  return 0;
}
