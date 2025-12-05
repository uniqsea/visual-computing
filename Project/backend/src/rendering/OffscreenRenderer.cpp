#include "rendering/OffscreenRenderer.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

// --- Minimal Math Library ---

struct Vec3 {
  float x, y, z;
  Vec3 operator+(const Vec3 &o) const { return {x + o.x, y + o.y, z + o.z}; }
  Vec3 operator-(const Vec3 &o) const { return {x - o.x, y - o.y, z - o.z}; }
  Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
  float dot(const Vec3 &o) const { return x * o.x + y * o.y + z * o.z; }
  Vec3 cross(const Vec3 &o) const {
    return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
  }
  Vec3 normalize() const {
    float len = std::sqrt(x * x + y * y + z * z);
    return len > 0 ? *this * (1.0f / len) : *this;
  }
};

struct Mat4 {
  float m[4][4] = {{0}};
  static Mat4 identity() {
    Mat4 res;
    for (int i = 0; i < 4; ++i)
      res.m[i][i] = 1.0f;
    return res;
  }
  static Mat4 lookAt(const Vec3 &eye, const Vec3 &center, const Vec3 &up) {
    Vec3 f = (center - eye).normalize();
    Vec3 s = f.cross(up).normalize();
    Vec3 u = s.cross(f);
    Mat4 res = identity();
    res.m[0][0] = s.x;
    res.m[0][1] = s.y;
    res.m[0][2] = s.z;
    res.m[1][0] = u.x;
    res.m[1][1] = u.y;
    res.m[1][2] = u.z;
    res.m[2][0] = -f.x;
    res.m[2][1] = -f.y;
    res.m[2][2] = -f.z;
    res.m[0][3] = -s.dot(eye);
    res.m[1][3] = -u.dot(eye);
    res.m[2][3] = f.dot(eye);
    return res;
  }
  static Mat4 perspective(float fov, float aspect, float near, float far) {
    float tanHalfFov = std::tan(fov / 2.0f);
    Mat4 res;
    res.m[0][0] = 1.0f / (aspect * tanHalfFov);
    res.m[1][1] = 1.0f / tanHalfFov;
    res.m[2][2] = -(far + near) / (far - near);
    res.m[2][3] = -(2.0f * far * near) / (far - near);
    res.m[3][2] = -1.0f;
    return res;
  }
  Vec3 transformPoint(const Vec3 &v) const {
    float x = m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3];
    float y = m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3];
    float z = m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3];
    float w = m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3];
    return w != 0 ? Vec3{x / w, y / w, z / w} : Vec3{x, y, z};
  }
  Mat4 operator*(const Mat4 &o) const {
    Mat4 res;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        res.m[i][j] = 0;
        for (int k = 0; k < 4; ++k)
          res.m[i][j] += m[i][k] * o.m[k][j];
      }
    }
    return res;
  }
};

// --- Rasterizer Helpers ---

float edgeFunction(const Vec3 &a, const Vec3 &b, const Vec3 &c) {
  return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

OffscreenRenderer::OffscreenRenderer() = default;

std::string
OffscreenRenderer::renderToImage(const Mesh &mesh, const std::string &outputDir,
                                 const RenderSettings &settings) const {
  std::filesystem::create_directories(outputDir);
  const int width = settings.width;
  const int height = settings.height;

  // Buffers
  std::vector<float> zBuffer(width * height, std::numeric_limits<float>::max());
  std::vector<Vec3> frameBuffer(width * height,
                                {0.1f, 0.1f, 0.15f}); // Background color

  // Matrices
  Vec3 eye = {2.0f, 2.0f, 2.0f};
  Vec3 center = {0.0f, 0.0f, 0.0f};
  Vec3 up = {0.0f, 1.0f, 0.0f};
  Mat4 view = Mat4::lookAt(eye, center, up);
  Mat4 proj = Mat4::perspective(1.047f, (float)width / height, 0.1f,
                                100.0f); // 60 deg FOV
  Mat4 vp = proj * view;

  // Light
  Vec3 lightDir = Vec3{1.0f, 1.0f, 1.0f}.normalize();

  // Rasterization Loop
  for (size_t i = 0; i < mesh.indices.size(); i += 3) {
    // Get vertices
    const auto &v0_raw = mesh.vertices[mesh.indices[i]];
    const auto &v1_raw = mesh.vertices[mesh.indices[i + 1]];
    const auto &v2_raw = mesh.vertices[mesh.indices[i + 2]];

    Vec3 v0_world = {v0_raw.position[0], v0_raw.position[1],
                     v0_raw.position[2]};
    Vec3 v1_world = {v1_raw.position[0], v1_raw.position[1],
                     v1_raw.position[2]};
    Vec3 v2_world = {v2_raw.position[0], v2_raw.position[1],
                     v2_raw.position[2]};

    // Transform to screen space
    Vec3 v0_proj = vp.transformPoint(v0_world);
    Vec3 v1_proj = vp.transformPoint(v1_world);
    Vec3 v2_proj = vp.transformPoint(v2_world);

    // Viewport transform
    auto toScreen = [&](const Vec3 &v) -> Vec3 {
      return {(v.x + 1) * 0.5f * width, (1 - (v.y + 1) * 0.5f) * height, v.z};
    };
    Vec3 v0 = toScreen(v0_proj);
    Vec3 v1 = toScreen(v1_proj);
    Vec3 v2 = toScreen(v2_proj);

    // Bounding box
    int minX = std::max(0, (int)std::min({v0.x, v1.x, v2.x}));
    int minY = std::max(0, (int)std::min({v0.y, v1.y, v2.y}));
    int maxX = std::min(width - 1, (int)std::max({v0.x, v1.x, v2.x}) + 1);
    int maxY = std::min(height - 1, (int)std::max({v0.y, v1.y, v2.y}) + 1);

    float area = edgeFunction(v0, v1, v2);

    for (int y = minY; y <= maxY; ++y) {
      for (int x = minX; x <= maxX; ++x) {
        Vec3 p = {(float)x + 0.5f, (float)y + 0.5f, 0};
        float w0 = edgeFunction(v1, v2, p);
        float w1 = edgeFunction(v2, v0, p);
        float w2 = edgeFunction(v0, v1, p);

        if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
          w0 /= area;
          w1 /= area;
          w2 /= area;
          float z = w0 * v0.z + w1 * v1.z + w2 * v2.z;

          if (z < zBuffer[y * width + x]) {
            zBuffer[y * width + x] = z;

            // Shading
            Vec3 n0 = {v0_raw.normal[0], v0_raw.normal[1], v0_raw.normal[2]};
            Vec3 n1 = {v1_raw.normal[0], v1_raw.normal[1], v1_raw.normal[2]};
            Vec3 n2 = {v2_raw.normal[0], v2_raw.normal[1], v2_raw.normal[2]};
            Vec3 normal = (n0 * w0 + n1 * w1 + n2 * w2).normalize();

            float diff = std::max(0.0f, normal.dot(lightDir));
            Vec3 color = {0.8f, 0.8f, 0.8f}; // Object color
            frameBuffer[y * width + x] =
                color * (diff * 0.8f + 0.2f); // Ambient + Diffuse
          }
        }
      }
    }
  }

  // Write to BMP
  const int rowStride = ((width * 3 + 3) / 4) * 4;
  std::vector<unsigned char> buffer(rowStride * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const Vec3 &c = frameBuffer[(height - 1 - y) * width + x];
      size_t index = y * rowStride + x * 3;
      buffer[index + 0] = (unsigned char)std::min(255.0f, c.z * 255); // B
      buffer[index + 1] = (unsigned char)std::min(255.0f, c.y * 255); // G
      buffer[index + 2] = (unsigned char)std::min(255.0f, c.x * 255); // R
    }
  }

  const std::string path = outputDir + "/render.bmp";
  const std::uint32_t pixelArraySize = rowStride * height;
  const std::uint32_t fileSize = 54 + pixelArraySize;
  unsigned char header[54] = {0};
  header[0] = 'B';
  header[1] = 'M';
  *(uint32_t *)&header[2] = fileSize;
  *(uint32_t *)&header[10] = 54;
  *(uint32_t *)&header[14] = 40;
  *(uint32_t *)&header[18] = width;
  *(uint32_t *)&header[22] = height;
  *(uint16_t *)&header[26] = 1;
  *(uint16_t *)&header[28] = 24;
  *(uint32_t *)&header[34] = pixelArraySize;

  std::ofstream file(path, std::ios::binary);
  file.write((char *)header, 54);
  file.write((char *)buffer.data(), buffer.size());
  return path;
}
