#pragma once

#include "geometry/Mesh.h"
#include <string>

class MeshSerializer {
public:
  std::string writeJson(const Mesh &mesh, const std::string &outputDir) const;
};
