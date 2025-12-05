#pragma once

#include "geometry/Mesh.h"
#include <string>
#include <vector>

struct SketchRequest {
  std::string mode;            // extrusion | revolution | heightmap
  std::string sketchPath;      // path to uploaded sketch
  std::vector<float> heightmap;
  float sketchThickness = 0.0f;
  float extrusionDepth = 0.25f;
  int extrusionSmoothSteps = 0;
  float extrusionBevelAmount = 0.0f;
  int revolutionSegments = 64;
  bool revolutionCapBottom = false;
  bool revolutionCapTop = false;
  float revolutionAxisOffsetX = 0.0f;
  bool revolutionHollow = false;
  float revolutionWallThickness = 0.05f;
  float revolutionAngleDegrees = 360.0f;
  float heightScale = 0.35f;
  bool heightWithBase = true;
  float heightBlurSigma = 0.0f;
  unsigned int heightResolution = 64;
  float heightBulgeStrength = 0.0f;
};

struct RenderResult {
  std::string renderImagePath;
  std::string meshJsonPath;
  std::string relativeDir;
};

class SketchProcessor;
class ExtrusionGenerator;
class RevolutionGenerator;
class HeightMapGenerator;
class OffscreenRenderer;
class MeshSerializer;

class PipelineController {
public:
  PipelineController();
  ~PipelineController();

  RenderResult handleRequest(const SketchRequest &request);
  const std::string &getOutputDirectory() const;

private:
  SketchProcessor *sketchProcessor;
  ExtrusionGenerator *extrusion;
  RevolutionGenerator *revolution;
  HeightMapGenerator *heightmap;
  OffscreenRenderer *renderer;
  MeshSerializer *serializer;

  std::string outputDir;
  std::string generateToken() const;
};
