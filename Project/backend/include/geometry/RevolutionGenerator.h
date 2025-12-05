#pragma once

#include "geometry/Mesh.h"
#include <vector>

struct RevolutionSettings {
  int segments = 64;
  bool capBottom = false;
  bool capTop = false;
  float axisOffsetX = 0.0f;     // horizontal shift of revolution axis
  bool hollow = false;          // generate inner tube surface
  float wallThickness = 0.05f;  // thickness in the same units as profile radius
  float angleDegrees = 360.0f;  // sweep angle
};

class RevolutionGenerator {
public:
  Mesh generate(const std::vector<std::array<float, 2>> &profile,
                const RevolutionSettings &settings) const;
};
