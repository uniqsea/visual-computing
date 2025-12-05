#pragma once

#include "geometry/Mesh.h"
#include <vector>

struct HeightMapSettings {
  unsigned int resolution = 64;
  float heightScale = 0.35f;
  bool withBase = true;
   // Approximate blur radius in texels for smoothing
  float blurSigma = 0.0f;
};

class HeightMapGenerator {
public:
  Mesh generate(const std::vector<float> &heightField,
                const HeightMapSettings &settings) const;
};
