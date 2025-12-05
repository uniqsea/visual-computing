#pragma once

#include <vector>
#include <array>

struct Vertex {
  std::array<float, 3> position{};
  std::array<float, 3> normal{};
};

struct Mesh {
  std::vector<Vertex> vertices;
  std::vector<unsigned int> indices;
  bool empty() const { return vertices.empty() || indices.empty(); }
};
