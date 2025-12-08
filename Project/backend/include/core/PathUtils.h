#pragma once

#include <filesystem>
#include <string>

namespace sketch3d {

// Returns the canonical data directory (prefers backend/data, falls back to data)
std::filesystem::path resolveDataRoot();

// Returns/creates a canonical subdirectory under the data directory
std::filesystem::path resolveDataSubdir(const std::string &name);

} // namespace sketch3d
