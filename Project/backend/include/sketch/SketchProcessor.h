#pragma once

#include <array>
#include <string>
#include <vector>

struct SketchData {
  std::vector<std::array<float, 2>> contour;
  std::vector<std::vector<std::array<float, 2>>> contours;
  std::vector<float> heightmap;
  std::vector<std::array<float, 2>> profile;
  int imageWidth = 576;
  int imageHeight = 640;
  float estimatedStroke = 2.0f;
};

class SketchProcessor {
public:
  SketchProcessor();
  SketchData process(const std::string &sketchPath, float strokeThickness,
                     float axisOffsetX = 0.0f) const;
  std::string exportToSvg(const SketchData &data,
                          int widthOverride = -1,
                          int heightOverride = -1) const;
  std::string vectorizeBitmapWithPotrace(const std::string &imagePath,
                                         float strokeThickness) const;
};
