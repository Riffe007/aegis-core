// The build's own identity: the commit it was made from and whether the tree
// had uncommitted tracked changes when it was built (cmake/build_info.cmake,
// regenerated on every build). Every result row carries it ("build_sha",
// "build_dirty"), and aegis-perf refuses to store results from a dirty or
// unknown build unless --allow-dirty (CLAUDE.md Benchmark policy).
#pragma once
#include <string>

namespace aegis::bench {

struct BuildInfo {
  const char* sha;  // 40 hex digits; "" if unknown (no git, or not a checkout)
  bool dirty;       // uncommitted tracked changes at build time (true if unknown)
  bool known;
};

extern const BuildInfo kBuild;

// `"build_sha":"<sha>","build_dirty":<bool>` (build_sha null if unknown), for a JSON row.
inline std::string build_json() {
  std::string sha = kBuild.known ? "\"" + std::string(kBuild.sha) + "\"" : "null";
  return "\"build_sha\":" + sha + ",\"build_dirty\":" + (kBuild.dirty ? "true" : "false");
}

}  // namespace aegis::bench
