#include "geometry/ExtrusionGenerator.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace {

float crossProduct(const std::array<float, 2> &a, const std::array<float, 2> &b,
                   const std::array<float, 2> &c) {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

bool isPointInTriangle(const std::array<float, 2> &p,
                       const std::array<float, 2> &a,
                       const std::array<float, 2> &b,
                       const std::array<float, 2> &c) {
  const float cp1 = crossProduct(a, b, p);
  const float cp2 = crossProduct(b, c, p);
  const float cp3 = crossProduct(c, a, p);
  return (cp1 >= 0 && cp2 >= 0 && cp3 >= 0) ||
         (cp1 <= 0 && cp2 <= 0 && cp3 <= 0);
}

bool isEar(const std::vector<std::array<float, 2>> &polygon,
           const std::vector<unsigned int> &indices, unsigned int u,
           unsigned int v, unsigned int w) {
  const auto &a = polygon[indices[u]];
  const auto &b = polygon[indices[v]];
  const auto &c = polygon[indices[w]];

  // Check if convex
  if (crossProduct(a, b, c) < 0) {
    return false;
  }

  // Check if any other point is inside
  for (unsigned int i = 0; i < indices.size(); ++i) {
    if (i == u || i == v || i == w) {
      continue;
    }
    if (isPointInTriangle(polygon[indices[i]], a, b, c)) {
      return false;
    }
  }
  return true;
}

std::vector<unsigned int>
triangulate(const std::vector<std::array<float, 2>> &contour) {
  std::vector<unsigned int> result;
  const auto n = contour.size();
  if (n < 3) {
    return result;
  }

  std::vector<unsigned int> indices(n);
  for (unsigned int i = 0; i < n; ++i) {
    indices[i] = i;
  }

  // Ensure counter-clockwise winding
  float area = 0.0f;
  for (unsigned int i = 0; i < n; ++i) {
    const auto &p1 = contour[i];
    const auto &p2 = contour[(i + 1) % n];
    area += (p2[0] - p1[0]) * (p2[1] + p1[1]);
  }
  if (area > 0) {
    std::reverse(indices.begin(), indices.end());
  }

  unsigned int count = n;
  unsigned int current = 0;
  while (count > 2) {
    bool found = false;
    for (unsigned int i = 0; i < count; ++i) {
      const unsigned int u = current % count;
      const unsigned int v = (current + 1) % count;
      const unsigned int w = (current + 2) % count;

      if (isEar(contour, indices, u, v, w)) {
        result.push_back(indices[u]);
        result.push_back(indices[v]);
        result.push_back(indices[w]);
        indices.erase(indices.begin() + v);
        count--;
        found = true;
        break;
      }
      current++;
    }
    if (!found) {
      // Fallback for degenerate cases
      break;
    }
  }
  return result;
}

std::vector<std::array<float, 2>>
subdivideContour(const std::vector<std::array<float, 2>> &contour,
                 int steps) {
  if (steps <= 0) {
    return contour;
  }
  std::vector<std::array<float, 2>> current = contour;
  for (int s = 0; s < steps; ++s) {
    std::vector<std::array<float, 2>> next;
    const size_t n = current.size();
    if (n < 2) {
      break;
    }
    for (size_t i = 0; i < n; ++i) {
      const auto &p0 = current[i];
      const auto &p1 = current[(i + 1) % n];
      next.push_back(p0);
      next.push_back({0.5f * (p0[0] + p1[0]), 0.5f * (p0[1] + p1[1])});
    }
    current = std::move(next);
  }
  return current;
}

} // namespace

Mesh ExtrusionGenerator::generate(
    const std::vector<std::array<float, 2>> &contourIn,
    const ExtrusionSettings &settings) const {
  Mesh mesh;
  if (contourIn.size() < 3) {
    return mesh;
  }

  const std::vector<std::array<float, 2>> contour =
      subdivideContour(contourIn, std::max(0, settings.smoothSteps));

  const float half = settings.depth * 0.5f;
  const auto count = static_cast<unsigned int>(contour.size());
  mesh.vertices.reserve(count * 4);

  std::array<float, 2> centroid{0.0f, 0.0f};
  for (const auto &p : contour) {
    centroid[0] += p[0];
    centroid[1] += p[1];
  }
  centroid[0] /= static_cast<float>(count);
  centroid[1] /= static_cast<float>(count);
  const float bevel = std::clamp(settings.bevelAmount, 0.0f, 0.9f);

  // Bottom cap vertices
  for (const auto &p : contour) {
    Vertex bottom;
    bottom.position = {p[0], p[1], -half};
    bottom.normal = {0.0f, 0.0f, -1.0f};
    mesh.vertices.push_back(bottom);
  }
  // Top cap vertices
  for (const auto &p : contour) {
    Vertex top;
    std::array<float, 2> tp = p;
    if (bevel > 0.0f) {
      tp[0] = centroid[0] + (p[0] - centroid[0]) * (1.0f - bevel);
      tp[1] = centroid[1] + (p[1] - centroid[1]) * (1.0f - bevel);
    }
    top.position = {tp[0], tp[1], half};
    top.normal = {0.0f, 0.0f, 1.0f};
    mesh.vertices.push_back(top);
  }

  // Triangulate caps
  std::vector<unsigned int> capIndices = triangulate(contour);

  // Bottom cap (reverse winding)
  for (size_t i = 0; i < capIndices.size(); i += 3) {
    mesh.indices.push_back(capIndices[i]);
    mesh.indices.push_back(capIndices[i + 2]);
    mesh.indices.push_back(capIndices[i + 1]);
  }

  // Top cap (offset indices)
  const unsigned int offset = count;
  for (unsigned int idx : capIndices) {
    mesh.indices.push_back(offset + idx);
  }

  // Side walls
  for (unsigned int i = 0; i < count; ++i) {
    const unsigned int next = (i + 1) % count;
    Vertex v0, v1, v2, v3;
    v0.position = {contour[i][0], contour[i][1], -half};
    v1.position = {contour[next][0], contour[next][1], -half};
    v2.position = {contour[next][0], contour[next][1], half};
    v3.position = {contour[i][0], contour[i][1], half};

    const float dx = contour[next][0] - contour[i][0];
    const float dy = contour[next][1] - contour[i][1];
    const float len = std::max(std::sqrt(dx * dx + dy * dy), 1e-5f);
    const float nx = dy / len;
    const float ny = -dx / len;

    v0.normal = {nx, ny, 0.0f};
    v1.normal = {nx, ny, 0.0f};
    v2.normal = {nx, ny, 0.0f};
    v3.normal = {nx, ny, 0.0f};

    const unsigned int base = static_cast<unsigned int>(mesh.vertices.size());
    mesh.vertices.insert(mesh.vertices.end(), {v0, v1, v2, v3});
    mesh.indices.insert(mesh.indices.end(),
                        {base, base + 1, base + 2, base, base + 2, base + 3});
  }
  return mesh;
}

Mesh ExtrusionGenerator::generate(
    const std::vector<std::vector<std::array<float, 2>>> &contours,
    const ExtrusionSettings &settings) const {
  Mesh combined;
  for (const auto &loop : contours) {
    Mesh part = generate(loop, settings);
    if (part.vertices.empty()) {
      continue;
    }
    const unsigned int base = static_cast<unsigned int>(combined.vertices.size());
    combined.vertices.insert(combined.vertices.end(), part.vertices.begin(),
                             part.vertices.end());
    for (unsigned int idx : part.indices) {
      combined.indices.push_back(base + idx);
    }
  }
  return combined;
}
