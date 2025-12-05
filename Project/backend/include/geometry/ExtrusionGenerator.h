#pragma once

#include "geometry/Mesh.h"
#include <vector>

struct ExtrusionSettings {
  float depth = 0.25f;
  int smoothSteps = 0;      // number of contour subdivision steps
  float bevelAmount = 0.0f; // 0..1, taper towards top
};

class ExtrusionGenerator {
public:
  Mesh generate(const std::vector<std::array<float, 2>> &contour,
                const ExtrusionSettings &settings) const;
  Mesh generate(const std::vector<std::vector<std::array<float, 2>>> &contours,
                const ExtrusionSettings &settings) const;
};
