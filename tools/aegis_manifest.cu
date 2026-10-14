// aegis_manifest: print a JSON description of the local GPU, including a
// MEASURED device-memory bandwidth. Rooflines use this number, not datasheets.
//
//   ./build/aegis_manifest > results/manifest-$(hostname).json
#include <cstdio>
#include <string>

#include "aegis/common.cuh"
#include "aegis/manifest.h"

int main() {
  int dev = 0;
  AEGIS_CUDA_CHECK(cudaGetDevice(&dev));
  std::string json;
  AEGIS_CUDA_CHECK(aegis::device_manifest_json(dev, json));
  std::fputs(json.c_str(), stdout);
  return 0;
}
