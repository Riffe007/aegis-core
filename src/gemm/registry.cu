// The single list of launchable GEMM impls. Tiers 0-1 are listed by hand;
// templated tiers contribute one entry per row of their config table.
#include <vector>

#include "aegis/gemm.h"

namespace aegis::gemm {

std::span<const KernelEntry> registry() {
  static const std::vector<KernelEntry> all = [] {
    std::vector<KernelEntry> v{
        {"naive", CONFIG_naive, &launch_naive, &attrs_naive},
        {"tiled", CONFIG_tiled, &launch_tiled, &attrs_tiled},
    };
    for (auto tier : {regblock_entries()}) v.insert(v.end(), tier.begin(), tier.end());
    return v;
  }();
  return all;
}

const KernelEntry* find(std::string_view name) {
  for (const KernelEntry& e : registry())
    if (name == e.name) return &e;
  return nullptr;
}

}  // namespace aegis::gemm
