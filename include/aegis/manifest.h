#pragma once
// Device manifest: a JSON description of a GPU, including a MEASURED
// device-memory bandwidth. Rooflines use this number, not datasheets.
// Shared by tools/aegis_manifest and the C ABI (aegis_device_manifest).

#include <cuda_runtime.h>

#include <string>

namespace aegis {

// Fills `json` for `device` (restores the caller's current device). Runs a
// ~1 GiB device-to-device copy benchmark; run with the system otherwise idle.
cudaError_t device_manifest_json(int device, std::string& json);

}  // namespace aegis
