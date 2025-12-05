#pragma once

#include "geometry/Mesh.h"
#include <string>

struct RenderSettings {
  int width = 1024;
  int height = 1024;
  bool writeDebugNormals = false;
};

class OffscreenRenderer {
public:
  OffscreenRenderer();
  std::string renderToImage(const Mesh &mesh, const std::string &outputDir,
                            const RenderSettings &settings) const;
};
