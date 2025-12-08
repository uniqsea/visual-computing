#include "core/PipelineController.h"
#include "core/PathUtils.h"
#include "geometry/ExtrusionGenerator.h"
#include "geometry/HeightMapGenerator.h"
#include "geometry/RevolutionGenerator.h"
#include "rendering/MeshSerializer.h"
#include "rendering/OffscreenRenderer.h"
#include "sketch/SketchProcessor.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <memory>
#include <queue>
#include <sstream>

namespace {

void applyHeightBulge(std::vector<float> &field, unsigned int resolution,
                      float strength) {
  if (strength <= 0.0f || field.empty() || resolution == 0) {
    return;
  }
  const size_t size = field.size();
  std::vector<int> distances(size, -1);
  std::queue<int> pending;
  auto index = [resolution](unsigned int x, unsigned int y) {
    return static_cast<int>(y * resolution + x);
  };
  bool hasZero = false;
  for (unsigned int y = 0; y < resolution; ++y) {
    for (unsigned int x = 0; x < resolution; ++x) {
      const int idx = index(x, y);
      if (field[idx] <= 0.0001f) {
        distances[idx] = 0;
        pending.push(idx);
        hasZero = true;
      }
    }
  }
  if (!hasZero) {
    for (unsigned int x = 0; x < resolution; ++x) {
      const int top = index(x, 0);
      if (distances[top] == -1) {
        distances[top] = 0;
        pending.push(top);
      }
      const int bottom = index(x, resolution - 1);
      if (distances[bottom] == -1) {
        distances[bottom] = 0;
        pending.push(bottom);
      }
    }
    for (unsigned int y = 0; y < resolution; ++y) {
      const int left = index(0, y);
      if (distances[left] == -1) {
        distances[left] = 0;
        pending.push(left);
      }
      const int right = index(resolution - 1, y);
      if (distances[right] == -1) {
        distances[right] = 0;
        pending.push(right);
      }
    }
  }
  const int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
  while (!pending.empty()) {
    const int current = pending.front();
    pending.pop();
    const unsigned int cx = static_cast<unsigned int>(current) % resolution;
    const unsigned int cy = static_cast<unsigned int>(current) / resolution;
    for (const auto &dir : dirs) {
      const int nx = static_cast<int>(cx) + dir[0];
      const int ny = static_cast<int>(cy) + dir[1];
      if (nx < 0 || ny < 0 || nx >= static_cast<int>(resolution) ||
          ny >= static_cast<int>(resolution)) {
        continue;
      }
      const int neighborIdx =
          index(static_cast<unsigned int>(nx), static_cast<unsigned int>(ny));
      if (distances[neighborIdx] == -1) {
        distances[neighborIdx] = distances[current] + 1;
        pending.push(neighborIdx);
      }
    }
  }
  int maxDistance = 0;
  for (size_t i = 0; i < size; ++i) {
    if (field[i] > 0.0001f && distances[i] > maxDistance) {
      maxDistance = distances[i];
    }
  }
  if (maxDistance <= 0) {
    return;
  }
  const float invMax = 1.0f / static_cast<float>(maxDistance);
  for (size_t i = 0; i < size; ++i) {
    if (field[i] > 0.0001f) {
      const float normalized =
          static_cast<float>(distances[i]) * invMax;
      const float value =
          (1.0f - strength) * field[i] + strength * normalized;
      field[i] = std::clamp(value, 0.0f, 1.0f);
    } else {
      field[i] = 0.0f;
    }
  }
}

} // namespace

PipelineController::PipelineController()
    : sketchProcessor(new SketchProcessor()), extrusion(new ExtrusionGenerator()),
      revolution(new RevolutionGenerator()), heightmap(new HeightMapGenerator()),
      renderer(new OffscreenRenderer()), serializer(new MeshSerializer()) {
  outputDir = sketch3d::resolveDataSubdir("outputs").string();
}

PipelineController::~PipelineController() {
  delete sketchProcessor;
  delete extrusion;
  delete revolution;
  delete heightmap;
  delete renderer;
  delete serializer;
}

RenderResult PipelineController::handleRequest(const SketchRequest &request) {
  SketchData data = sketchProcessor->process(request.sketchPath,
                                             request.sketchThickness,
                                             request.revolutionAxisOffsetX);
  Mesh mesh;
  if (request.mode == "revolution") {
    RevolutionSettings settings;
    if (request.revolutionSegments > 0) {
      settings.segments = request.revolutionSegments;
    }
    settings.capBottom = request.revolutionCapBottom;
    settings.capTop = request.revolutionCapTop;
    settings.axisOffsetX = request.revolutionAxisOffsetX;
    settings.hollow = request.revolutionHollow;
    settings.wallThickness = request.revolutionWallThickness;
    settings.angleDegrees = request.revolutionAngleDegrees;
    mesh = revolution->generate(data.profile, settings);
  } else if (request.mode == "heightmap") {
    std::vector<float> field =
        !data.heightmap.empty() ? data.heightmap : request.heightmap;
    unsigned int srcRes =
        static_cast<unsigned int>(std::sqrt(field.size()));
    if (srcRes * srcRes != field.size()) {
      srcRes = 0;
    }
    const unsigned int targetRes =
        request.heightResolution > 0 ? request.heightResolution : srcRes;
    std::vector<float> resampled = field;
    if (srcRes > 0 && targetRes > 0 && targetRes != srcRes) {
      resampled.assign(targetRes * targetRes, 0.0f);
      for (unsigned int y = 0; y < targetRes; ++y) {
        for (unsigned int x = 0; x < targetRes; ++x) {
          const float gx =
              static_cast<float>(x) / static_cast<float>(targetRes - 1);
          const float gy =
              static_cast<float>(y) / static_cast<float>(targetRes - 1);
          const float sx = gx * static_cast<float>(srcRes - 1);
          const float sy = gy * static_cast<float>(srcRes - 1);
          const unsigned int x0 = static_cast<unsigned int>(std::floor(sx));
          const unsigned int y0 = static_cast<unsigned int>(std::floor(sy));
          const unsigned int x1 = std::min(x0 + 1, srcRes - 1);
          const unsigned int y1 = std::min(y0 + 1, srcRes - 1);
          const float tx = sx - static_cast<float>(x0);
          const float ty = sy - static_cast<float>(y0);
          const float h00 = field[y0 * srcRes + x0];
          const float h10 = field[y0 * srcRes + x1];
          const float h01 = field[y1 * srcRes + x0];
          const float h11 = field[y1 * srcRes + x1];
          const float h0 = h00 * (1.0f - tx) + h10 * tx;
          const float h1 = h01 * (1.0f - tx) + h11 * tx;
          resampled[y * targetRes + x] = h0 * (1.0f - ty) + h1 * ty;
        }
      }
    }
    if (!resampled.empty() && targetRes > 0 &&
        request.heightBulgeStrength > 0.0f) {
      applyHeightBulge(resampled, targetRes, request.heightBulgeStrength);
    }

    HeightMapSettings settings;
    settings.resolution =
        targetRes > 0 ? targetRes : settings.resolution;
    if (request.heightScale > 0.0f) {
      settings.heightScale = request.heightScale;
    }
    settings.withBase = request.heightWithBase;
    settings.blurSigma = request.heightBlurSigma;
    mesh = heightmap->generate(resampled, settings);
  } else {
    ExtrusionSettings settings;
    if (request.extrusionDepth > 0.0f) {
      settings.depth = request.extrusionDepth;
    }
    settings.smoothSteps = std::max(0, request.extrusionSmoothSteps);
    settings.bevelAmount = 0.0f;
    std::vector<std::vector<std::array<float, 2>>> loops = data.contours;
    if (loops.empty() && !data.contour.empty()) {
      loops.push_back(data.contour);
    }
    mesh = extrusion->generate(loops, settings);
  }

  const std::string token = generateToken();
  std::filesystem::path baseDir =
      std::filesystem::path(outputDir) / token;
  std::filesystem::create_directories(baseDir);

  RenderResult result;
  result.relativeDir = token;
  result.meshJsonPath = serializer->writeJson(mesh, baseDir.string());
  result.renderImagePath =
      renderer->renderToImage(mesh, baseDir.string(), RenderSettings{});
  return result;
}

const std::string &PipelineController::getOutputDirectory() const {
  return outputDir;
}

std::string PipelineController::generateToken() const {
  using namespace std::chrono;
  auto now = time_point_cast<milliseconds>(system_clock::now()).time_since_epoch().count();
  std::stringstream ss;
  ss << now;
  return ss.str();
}
