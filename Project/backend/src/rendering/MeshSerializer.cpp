#include "rendering/MeshSerializer.h"
#include <filesystem>
#include <fstream>

std::string MeshSerializer::writeJson(const Mesh &mesh,
                                      const std::string &outputDir) const {
  std::filesystem::create_directories(outputDir);
  const std::string path = outputDir + "/mesh.json";
  std::ofstream file(path);
  file << "{\n  \"positions\": [";
  for (size_t i = 0; i < mesh.vertices.size(); ++i) {
    const auto &pos = mesh.vertices[i].position;
    file << pos[0] << "," << pos[1] << "," << pos[2];
    if (i + 1 < mesh.vertices.size()) {
      file << ",";
    }
  }
  file << "],\n  \"normals\": [";
  for (size_t i = 0; i < mesh.vertices.size(); ++i) {
    const auto &n = mesh.vertices[i].normal;
    file << n[0] << "," << n[1] << "," << n[2];
    if (i + 1 < mesh.vertices.size()) {
      file << ",";
    }
  }
  file << "],\n  \"indices\": [";
  for (size_t i = 0; i < mesh.indices.size(); ++i) {
    file << mesh.indices[i];
    if (i + 1 < mesh.indices.size()) {
      file << ",";
    }
  }
  file << "]\n}\n";
  return path;
}
