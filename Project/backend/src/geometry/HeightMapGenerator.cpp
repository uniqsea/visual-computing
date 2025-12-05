#include "geometry/HeightMapGenerator.h"
#include <algorithm>

Mesh HeightMapGenerator::generate(const std::vector<float> &heightField,
                                  const HeightMapSettings &settings) const {
  Mesh mesh;
  const unsigned int res = settings.resolution;
  if (heightField.size() != res * res) {
    return mesh;
  }

  std::vector<float> smoothed = heightField;
  if (settings.blurSigma > 0.0f) {
    const int radius =
        std::max(1, static_cast<int>(settings.blurSigma * static_cast<float>(res)));
    std::vector<float> temp(res * res, 0.0f);
    for (unsigned int y = 0; y < res; ++y) {
      for (unsigned int x = 0; x < res; ++x) {
        float acc = 0.0f;
        int count = 0;
        for (int k = -radius; k <= radius; ++k) {
          const int xx = static_cast<int>(x) + k;
          if (xx >= 0 && xx < static_cast<int>(res)) {
            acc += smoothed[y * res + xx];
            ++count;
          }
        }
        temp[y * res + x] =
            (count > 0) ? acc / static_cast<float>(count) : 0.0f;
      }
    }
    smoothed.swap(temp);
    for (unsigned int x = 0; x < res; ++x) {
      for (unsigned int y = 0; y < res; ++y) {
        float acc = 0.0f;
        int count = 0;
        for (int k = -radius; k <= radius; ++k) {
          const int yy = static_cast<int>(y) + k;
          if (yy >= 0 && yy < static_cast<int>(res)) {
            acc += smoothed[yy * res + x];
            ++count;
          }
        }
        temp[y * res + x] =
            (count > 0) ? acc / static_cast<float>(count) : 0.0f;
      }
    }
    smoothed.swap(temp);
  }

  if (settings.withBase) {
    mesh.vertices.reserve(res * res);
    for (unsigned int y = 0; y < res; ++y) {
      for (unsigned int x = 0; x < res; ++x) {
        const float fx = static_cast<float>(x) / static_cast<float>(res - 1);
        const float fy = static_cast<float>(y) / static_cast<float>(res - 1);
        Vertex v;
        v.position = {fx - 0.5f,
                      smoothed[y * res + x] * settings.heightScale,
                      fy - 0.5f};
        v.normal = {0.0f, 1.0f, 0.0f};
        mesh.vertices.push_back(v);
      }
    }
    for (unsigned int y = 0; y + 1 < res; ++y) {
      for (unsigned int x = 0; x + 1 < res; ++x) {
        const unsigned int v0 = y * res + x;
        const unsigned int v1 = y * res + x + 1;
        const unsigned int v2 = (y + 1) * res + x;
        const unsigned int v3 = (y + 1) * res + x + 1;
        mesh.indices.insert(mesh.indices.end(), {v0, v1, v2, v1, v3, v2});
      }
    }
    return mesh;
  }

  const float eps = 1e-4f;
  for (unsigned int y = 0; y + 1 < res; ++y) {
    for (unsigned int x = 0; x + 1 < res; ++x) {
      const float h00 = smoothed[y * res + x];
      const float h10 = smoothed[y * res + x + 1];
      const float h01 = smoothed[(y + 1) * res + x];
      const float h11 = smoothed[(y + 1) * res + x + 1];
      const float maxH =
          std::max(std::max(h00, h10), std::max(h01, h11));
      if (maxH <= eps) {
        continue;
      }
      const float fx0 = static_cast<float>(x) / static_cast<float>(res - 1);
      const float fx1 =
          static_cast<float>(x + 1) / static_cast<float>(res - 1);
      const float fy0 = static_cast<float>(y) / static_cast<float>(res - 1);
      const float fy1 =
          static_cast<float>(y + 1) / static_cast<float>(res - 1);

      const unsigned int base =
          static_cast<unsigned int>(mesh.vertices.size());
      Vertex v0, v1, v2, v3;
      v0.position = {fx0 - 0.5f, h00 * settings.heightScale, fy0 - 0.5f};
      v1.position = {fx1 - 0.5f, h10 * settings.heightScale, fy0 - 0.5f};
      v2.position = {fx0 - 0.5f, h01 * settings.heightScale, fy1 - 0.5f};
      v3.position = {fx1 - 0.5f, h11 * settings.heightScale, fy1 - 0.5f};
      v0.normal = v1.normal = v2.normal = v3.normal = {0.0f, 1.0f, 0.0f};
      mesh.vertices.push_back(v0);
      mesh.vertices.push_back(v1);
      mesh.vertices.push_back(v2);
      mesh.vertices.push_back(v3);
      mesh.indices.insert(mesh.indices.end(),
                          {base, base + 1, base + 2, base + 1, base + 3,
                           base + 2});
    }
  }
  return mesh;
}
