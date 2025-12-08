#include "core/PathUtils.h"
#include <array>

namespace {
std::filesystem::path resolveCandidate(const std::filesystem::path &candidate) {
  std::error_code ec;
  if (std::filesystem::exists(candidate, ec)) {
    return std::filesystem::weakly_canonical(candidate, ec);
  }
  return {};
}
} // namespace

namespace sketch3d {

std::filesystem::path resolveDataRoot() {
  static std::filesystem::path cached;
  if (!cached.empty()) {
    return cached;
  }

  const std::array<std::filesystem::path, 2> candidates = {
      std::filesystem::path("backend") / "data",
      std::filesystem::path("data")};

  for (const auto &candidate : candidates) {
    auto resolved = resolveCandidate(candidate);
    if (!resolved.empty()) {
      cached = resolved;
      return cached;
    }
  }

  // Neither path existed – create the first candidate relative to CWD
  std::filesystem::path fallback =
      std::filesystem::current_path() / candidates.front();
  std::error_code ec;
  std::filesystem::create_directories(fallback, ec);
  cached = resolveCandidate(fallback);
  if (cached.empty()) {
    cached = fallback;
  }
  return cached;
}

std::filesystem::path resolveDataSubdir(const std::string &name) {
  auto root = resolveDataRoot();
  auto subdir = root / name;
  std::error_code ec;
  std::filesystem::create_directories(subdir, ec);
  auto resolved = resolveCandidate(subdir);
  return resolved.empty() ? subdir : resolved;
}

} // namespace sketch3d
