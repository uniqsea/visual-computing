#include "geometry/RevolutionGenerator.h"
#include <array>
#include <cmath>
#include <vector>

Mesh RevolutionGenerator::generate(
    const std::vector<std::array<float, 2>> &profile,
    const RevolutionSettings &settings) const {
  Mesh mesh;
  if (profile.size() < 2 || settings.segments < 3) {
    return mesh;
  }
  const unsigned int segments = static_cast<unsigned int>(settings.segments);
  const unsigned int ring = static_cast<unsigned int>(profile.size());
  const float angleRad =
      std::max(0.0f, std::min(settings.angleDegrees, 360.0f)) *
      (3.1415926f / 180.0f);
  const float axisOffsetX = settings.axisOffsetX;

  const bool hollow = settings.hollow;
  const float wall = settings.wallThickness;

  for (unsigned int seg = 0; seg <= segments; ++seg) {
    const float t = static_cast<float>(seg) / static_cast<float>(segments);
    const float angle = t * (angleRad > 0.0f ? angleRad : 2.0f * 3.1415926f);
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    for (const auto &p : profile) {
      Vertex v;
      const float r = std::max(0.0f, p[0]);
      v.position = {axisOffsetX + r * c, p[1], r * s};
      v.normal = {c, 0.0f, s};
      mesh.vertices.push_back(v);

      if (hollow) {
        Vertex inner;
        const float innerR = std::max(0.0f, r - wall);
        inner.position = {axisOffsetX + innerR * c, p[1], innerR * s};
        inner.normal = {-c, 0.0f, -s};
        mesh.vertices.push_back(inner);
      }
    }
  }

  const unsigned int stride = hollow ? 2 * ring : ring;
  for (unsigned int seg = 0; seg < segments; ++seg) {
    const unsigned int nextSeg = seg + 1;
    for (unsigned int i = 0; i + 1 < ring; ++i) {
      const unsigned int a = seg * stride + (hollow ? 2 * i : i);
      const unsigned int b = nextSeg * stride + (hollow ? 2 * i : i);
      const unsigned int aNext =
          seg * stride + (hollow ? 2 * (i + 1) : (i + 1));
      const unsigned int bNext =
          nextSeg * stride + (hollow ? 2 * (i + 1) : (i + 1));
      mesh.indices.insert(mesh.indices.end(),
                          {a, aNext, b, b, aNext, bNext});
      if (hollow) {
        const unsigned int ai = a + 1;
        const unsigned int bi = b + 1;
        const unsigned int aNexti = aNext + 1;
        const unsigned int bNexti = bNext + 1;
        mesh.indices.insert(
            mesh.indices.end(),
            {ai, bi, aNexti, bi, bNexti, aNexti});
      }
    }
  }

  if ((settings.capBottom || settings.capTop) && ring >= 1) {
    const bool frontIsLower = profile.front()[1] <= profile.back()[1];
    const float yBottom =
        frontIsLower ? profile.front()[1] : profile.back()[1];
    const float yTop = frontIsLower ? profile.back()[1] : profile.front()[1];
    const unsigned int bottomIdxLocal = frontIsLower ? 0 : (ring - 1);
    const unsigned int topIdxLocal = frontIsLower ? (ring - 1) : 0;
    const unsigned int bottomStrideOffset =
        hollow ? 2 * bottomIdxLocal : bottomIdxLocal;
    const unsigned int topStrideOffset =
        hollow ? 2 * topIdxLocal : topIdxLocal;

    unsigned int bottomCenterIndex = 0;
    unsigned int topCenterIndex = 0;

    if (settings.capBottom) {
      Vertex bottomCenter;
      bottomCenter.position = {axisOffsetX, yBottom, 0.0f};
      bottomCenter.normal = {0.0f, -1.0f, 0.0f};
      bottomCenterIndex = static_cast<unsigned int>(mesh.vertices.size());
      mesh.vertices.push_back(bottomCenter);
    }

    if (settings.capTop) {
      Vertex topCenter;
      topCenter.position = {axisOffsetX, yTop, 0.0f};
      topCenter.normal = {0.0f, 1.0f, 0.0f};
      topCenterIndex = static_cast<unsigned int>(mesh.vertices.size());
      mesh.vertices.push_back(topCenter);
    }

    for (unsigned int seg = 0; seg < segments; ++seg) {
      const unsigned int nextSeg = seg + 1;
      if (settings.capBottom) {
        const unsigned int a = seg * stride + bottomStrideOffset;
        const unsigned int b = nextSeg * stride + bottomStrideOffset;
        mesh.indices.insert(mesh.indices.end(), {bottomCenterIndex, b, a});
      }

      if (settings.capTop) {
        const unsigned int aTop = seg * stride + topStrideOffset;
        const unsigned int bTop = nextSeg * stride + topStrideOffset;
        mesh.indices.insert(mesh.indices.end(), {topCenterIndex, aTop, bTop});
      }
    }
  }
  return mesh;
}
