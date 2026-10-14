#pragma once
// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 (PTX ISA 9.4 9.7.16.5.14,
// sm_80+), the instruction the tier 3/4 GEMMs issue.
#include <cstdint>

__device__ __forceinline__ void mma_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                          uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

constexpr double kMmaFlops = 2.0 * 16 * 8 * 16;  // per warp-level m16n8k16
