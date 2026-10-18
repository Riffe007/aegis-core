// aegis_hw: hardware discovery (include/aegis/hw.h) as JSON on stdout.
//
//   aegis_hw [--device D] [--root DIR] [--no-cuda] [--no-nvml] [--no-video] [--no-auxv]
//
// --root reads sysfs / procfs / dev under DIR (a recorded fixture, tests/fixtures/hw/);
// it implies --no-auxv (ISA from DIR/proc/cpuinfo). aegis-perf hw runs this.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "aegis/hw.h"

int main(int argc, char** argv) {
  aegis::hw::DiscoverOptions opt;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--device" && i + 1 < argc) opt.device = std::atoi(argv[++i]);
    else if (a == "--root" && i + 1 < argc) opt.root = argv[++i];
    else if (a == "--no-cuda") opt.cuda = false;
    else if (a == "--no-nvml") opt.nvml = false;
    else if (a == "--no-video") opt.video = false;
    else if (a == "--no-auxv") opt.auxv = false;
    else {
      std::fprintf(stderr, "usage: aegis_hw [--device D] [--root DIR] [--no-cuda] [--no-nvml] [--no-video] [--no-auxv]\n");
      return 2;
    }
  }
  std::fputs(aegis::hw::to_json(aegis::hw::discover(opt)).c_str(), stdout);
  return 0;
}
