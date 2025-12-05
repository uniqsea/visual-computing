#include "sketch/SketchProcessor.h"
#if __has_include(<opencv2/opencv.hpp>)
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool hasSvgExtension(const std::string &path) {
  if (path.size() < 4) {
    return false;
  }
  std::string lower = path;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return lower.rfind(".svg") == lower.size() - 4;
}

struct SvgShape {
  std::vector<std::array<float, 2>> points;
  bool closed = false;
  float strokeWidth = 2.0f;
};

struct SvgDocument {
  int width = 576;
  int height = 640;
  std::vector<SvgShape> shapes;
};

struct SvgTransform {
  float a = 1.0f;
  float b = 0.0f;
  float c = 0.0f;
  float d = 1.0f;
  float e = 0.0f;
  float f = 0.0f;
};

constexpr float kPi = 3.14159265358979323846f;

SvgTransform multiplyTransform(const SvgTransform &lhs,
                               const SvgTransform &rhs) {
  SvgTransform result;
  result.a = lhs.a * rhs.a + lhs.c * rhs.b;
  result.b = lhs.b * rhs.a + lhs.d * rhs.b;
  result.c = lhs.a * rhs.c + lhs.c * rhs.d;
  result.d = lhs.b * rhs.c + lhs.d * rhs.d;
  result.e = lhs.a * rhs.e + lhs.c * rhs.f + lhs.e;
  result.f = lhs.b * rhs.e + lhs.d * rhs.f + lhs.f;
  return result;
}

void applyTransform(const SvgTransform &transform,
                    std::vector<std::array<float, 2>> &points) {
  for (auto &pt : points) {
    const float x = pt[0];
    const float y = pt[1];
    pt[0] = transform.a * x + transform.c * y + transform.e;
    pt[1] = transform.b * x + transform.d * y + transform.f;
  }
}

std::vector<float> parseFloatList(const std::string &text) {
  std::vector<float> values;
  const char *ptr = text.c_str();
  while (*ptr) {
    while (*ptr &&
           (std::isspace(static_cast<unsigned char>(*ptr)) || *ptr == ',')) {
      ++ptr;
    }
    if (!*ptr) {
      break;
    }
    char *end = nullptr;
    float value = std::strtof(ptr, &end);
    if (end == ptr) {
      ++ptr;
      continue;
    }
    values.push_back(value);
    ptr = end;
  }
  return values;
}

bool parseTransformList(const std::string &text, SvgTransform &out) {
  if (text.empty()) {
    return false;
  }
  SvgTransform current;
  bool applied = false;
  size_t pos = 0;
  while (pos < text.size()) {
    while (pos < text.size() &&
           std::isspace(static_cast<unsigned char>(text[pos]))) {
      ++pos;
    }
    size_t nameStart = pos;
    while (pos < text.size() &&
           std::isalpha(static_cast<unsigned char>(text[pos]))) {
      ++pos;
    }
    if (nameStart == pos) {
      if (pos < text.size()) {
        ++pos;
      }
      continue;
    }
    std::string name = text.substr(nameStart, pos - nameStart);
    while (pos < text.size() &&
           std::isspace(static_cast<unsigned char>(text[pos]))) {
      ++pos;
    }
    if (pos >= text.size() || text[pos] != '(') {
      break;
    }
    ++pos;
    int depth = 1;
    size_t argsStart = pos;
    while (pos < text.size() && depth > 0) {
      if (text[pos] == '(') {
        ++depth;
      } else if (text[pos] == ')') {
        --depth;
      }
      ++pos;
    }
    if (depth != 0) {
      break;
    }
    const size_t argsEnd = pos - 1;
    std::string argsText = text.substr(argsStart, argsEnd - argsStart);
    std::vector<float> args = parseFloatList(argsText);
    SvgTransform local;
    if (name == "translate") {
      if (!args.empty()) {
        local.e = args[0];
        local.f = args.size() > 1 ? args[1] : 0.0f;
      }
    } else if (name == "scale") {
      if (!args.empty()) {
        local.a = args[0];
        local.d = args.size() > 1 ? args[1] : args[0];
      }
    } else if (name == "matrix" && args.size() == 6) {
      local.a = args[0];
      local.b = args[1];
      local.c = args[2];
      local.d = args[3];
      local.e = args[4];
      local.f = args[5];
    } else if (name == "rotate" && !args.empty()) {
      const float angle = args[0] * (kPi / 180.0f);
      const float cosA = std::cos(angle);
      const float sinA = std::sin(angle);
      SvgTransform rotation;
      rotation.a = cosA;
      rotation.b = sinA;
      rotation.c = -sinA;
      rotation.d = cosA;
      rotation.e = 0.0f;
      rotation.f = 0.0f;
      if (args.size() >= 3) {
        SvgTransform translateTo;
        translateTo.e = args[1];
        translateTo.f = args[2];
        SvgTransform translateBack;
        translateBack.e = -args[1];
        translateBack.f = -args[2];
        local = multiplyTransform(translateTo,
                                  multiplyTransform(rotation, translateBack));
      } else {
        local = rotation;
      }
    } else {
      continue;
    }
    current = multiplyTransform(local, current);
    applied = true;
  }
  if (applied) {
    out = current;
  }
  return applied;
}

std::string getAttributeValue(const std::string &tag,
                              const std::string &attribute) {
  std::string key = attribute + "=";
  size_t pos = 0;
  while ((pos = tag.find(key, pos)) != std::string::npos) {
    if (pos > 0) {
      char prev = tag[pos - 1];
      if (std::isalnum(static_cast<unsigned char>(prev)) || prev == '-' ||
          prev == ':') {
        pos += key.size();
        continue;
      }
    }
    size_t start = pos + key.size();
    while (start < tag.size() &&
           std::isspace(static_cast<unsigned char>(tag[start]))) {
      ++start;
    }
    if (start >= tag.size()) {
      break;
    }
    char quote = tag[start];
    if (quote == '\"' || quote == '\'') {
      ++start;
      size_t end = tag.find(quote, start);
      if (end == std::string::npos) {
        break;
      }
      return tag.substr(start, end - start);
    } else {
      size_t end = start;
      while (end < tag.size() &&
             !std::isspace(static_cast<unsigned char>(tag[end])) &&
             tag[end] != '/') {
        ++end;
      }
      return tag.substr(start, end - start);
    }
  }
  return "";
}

void normalizeSvgCoordinates(SvgDocument &doc) {
  if (doc.shapes.empty()) {
    return;
  }
  float minX = std::numeric_limits<float>::max();
  float maxX = std::numeric_limits<float>::lowest();
  float minY = std::numeric_limits<float>::max();
  float maxY = std::numeric_limits<float>::lowest();
  bool hasPoints = false;
  for (const auto &shape : doc.shapes) {
    for (const auto &pt : shape.points) {
      minX = std::min(minX, pt[0]);
      maxX = std::max(maxX, pt[0]);
      minY = std::min(minY, pt[1]);
      maxY = std::max(maxY, pt[1]);
      hasPoints = true;
    }
  }
  if (!hasPoints) {
    return;
  }
  const float widthF = static_cast<float>(doc.width);
  const float heightF = static_cast<float>(doc.height);
  const bool needScaleX = minX < -0.01f || maxX > widthF + 0.01f;
  const bool needScaleY = minY < -0.01f || maxY > heightF + 0.01f;
  if (!needScaleX && !needScaleY) {
    return;
  }
  const float epsilon = 1e-3f;
  const float denomX = std::max(maxX - minX, epsilon);
  const float denomY = std::max(maxY - minY, epsilon);
  const float scaleX = needScaleX ? widthF / denomX : 1.0f;
  const float scaleY = needScaleY ? heightF / denomY : 1.0f;
  const float clampMaxX = std::max(widthF - 1.0f, 0.0f);
  const float clampMaxY = std::max(heightF - 1.0f, 0.0f);
  for (auto &shape : doc.shapes) {
    for (auto &pt : shape.points) {
      if (needScaleX) {
        pt[0] = std::clamp((pt[0] - minX) * scaleX, 0.0f, clampMaxX);
      }
      if (needScaleY) {
        pt[1] = std::clamp((pt[1] - minY) * scaleY, 0.0f, clampMaxY);
      }
    }
  }
}

bool parseSvgPath(const std::string &d, std::vector<SvgShape> &shapes) {
  auto isNumberStart = [](int c) -> bool {
    return c == '-' || c == '+' || c == '.' ||
           (c >= '0' && c <= '9');
  };
  std::stringstream ss(d);
  char cmd;
  float args[6];
  std::array<float, 2> current = {0.0f, 0.0f};
  std::array<float, 2> start = {0.0f, 0.0f};
  SvgShape currentShape;
  currentShape.strokeWidth = 2.0f;

  auto finalizeShape = [&]() {
    if (!currentShape.points.empty()) {
      shapes.push_back(currentShape);
      currentShape = SvgShape{};
      currentShape.strokeWidth = 2.0f;
    }
  };

  // Simple parser for M, L, C, Z (absolute) and m, l, c, z (relative).
  // Supports implicit coordinate repetitions as emitted by Potrace.
  while (ss >> cmd) {
    if (cmd == 'M' || cmd == 'm') {
      bool firstPair = true;
      while (true) {
        if (!(ss >> args[0] >> args[1]))
          break;
        char effective = firstPair ? cmd : (cmd == 'M' ? 'L' : 'l');
        if (effective == 'm' || effective == 'l') {
          args[0] += current[0];
          args[1] += current[1];
        }
        current = {args[0], args[1]};
        if (firstPair) {
          start = current;
          if (!currentShape.points.empty()) {
            finalizeShape();
          }
          currentShape.points.push_back(current);
          currentShape.closed = false;
          firstPair = false;
        } else {
          currentShape.points.push_back(current);
        }
        ss >> std::ws;
        if (!isNumberStart(ss.peek()))
          break;
      }
    } else if (cmd == 'L' || cmd == 'l') {
      while (true) {
        if (!(ss >> args[0] >> args[1]))
          break;
        if (cmd == 'l') {
          args[0] += current[0];
          args[1] += current[1];
        }
        current = {args[0], args[1]};
        currentShape.points.push_back(current);
        ss >> std::ws;
        if (!isNumberStart(ss.peek()))
          break;
      }
    } else if (cmd == 'C' || cmd == 'c') {
      while (true) {
        if (!(ss >> args[0] >> args[1] >> args[2] >> args[3] >> args[4] >>
              args[5]))
          break;
        if (cmd == 'c') {
          args[0] += current[0];
          args[1] += current[1];
          args[2] += current[0];
          args[3] += current[1];
          args[4] += current[0];
          args[5] += current[1];
        }
        // Sample cubic bezier
        const int segments = 10;
        for (int i = 1; i <= segments; ++i) {
          float t = static_cast<float>(i) / segments;
          float t1 = 1.0f - t;
          float x = t1 * t1 * t1 * current[0] + 3 * t1 * t1 * t * args[0] +
                    3 * t1 * t * t * args[2] + t * t * t * args[4];
          float y = t1 * t1 * t1 * current[1] + 3 * t1 * t1 * t * args[1] +
                    3 * t1 * t * t * args[3] + t * t * t * args[5];
          currentShape.points.push_back({x, y});
        }
        current = {args[4], args[5]};
        ss >> std::ws;
        if (!isNumberStart(ss.peek()))
          break;
      }
    } else if (cmd == 'Z' || cmd == 'z') {
      currentShape.closed = true;
      // Close loop if needed
      if (!currentShape.points.empty() &&
          (std::abs(current[0] - start[0]) > 0.1f ||
           std::abs(current[1] - start[1]) > 0.1f)) {
        currentShape.points.push_back(start);
      }
      current = start;
      finalizeShape();
    } else {
      // Skip unknown or comma
      if (cmd != ',') {
        // Try to read as number if implicit command (e.g. L x y x y)
        // For simplicity, we assume explicit commands for now, or just skip.
      }
    }
  }
  finalizeShape();
  return !shapes.empty();
}

bool loadSvgDocument(const std::string &path, SvgDocument &doc) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return false;
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  const std::string content = buffer.str();

  // 1. Try to load custom metadata
  const std::string marker = "<metadata id=\"sketch-data\">";
  const auto metaStart = content.find(marker);
  if (metaStart != std::string::npos) {
    const auto metaEnd = content.find("</metadata>", metaStart);
    if (metaEnd != std::string::npos) {
      const auto jsonStart = metaStart + marker.size();
      const std::string jsonText =
          content.substr(jsonStart, metaEnd - jsonStart);
      try {
        auto meta = nlohmann::json::parse(jsonText);
        doc.width = meta.value("width", 576);
        doc.height = meta.value("height", 640);
        doc.shapes.clear();
        for (const auto &shape : meta["shapes"]) {
          SvgShape s;
          s.closed = shape.value("closed", false);
          s.strokeWidth = shape.value("strokeWidth", 2.0f);
          for (const auto &pt : shape["points"]) {
            if (pt.size() == 2) {
              s.points.push_back({pt[0].get<float>(), pt[1].get<float>()});
            }
          }
          if (!s.points.empty()) {
            doc.shapes.push_back(std::move(s));
          }
        }
        return true;
      } catch (const std::exception &) {
        // Fallback to standard parsing
      }
    }
  }

  // 2. Fallback: Parse standard SVG with minimal tag handling
  auto updateDimensions = [&](const std::string &tag) {
    std::string w = getAttributeValue(tag, "width");
    std::string h = getAttributeValue(tag, "height");
    if (!w.empty()) {
      doc.width = std::atoi(w.c_str());
    }
    if (!h.empty()) {
      doc.height = std::atoi(h.c_str());
    }
  };
  size_t svgPos = content.find("<svg");
  if (svgPos != std::string::npos) {
    size_t svgEnd = content.find('>', svgPos);
    if (svgEnd != std::string::npos) {
      std::string svgTag = content.substr(svgPos + 4, svgEnd - svgPos - 4);
      updateDimensions(svgTag);
    }
  }

  doc.shapes.clear();
  std::vector<SvgTransform> transformStack;
  transformStack.push_back(SvgTransform{});
  size_t pos = 0;
  while (pos < content.size()) {
    size_t tagStart = content.find('<', pos);
    if (tagStart == std::string::npos) {
      break;
    }
    if (content.compare(tagStart, 4, "<!--") == 0) {
      size_t commentEnd = content.find("-->", tagStart + 4);
      if (commentEnd == std::string::npos) {
        break;
      }
      pos = commentEnd + 3;
      continue;
    }
    if (content.compare(tagStart, 2, "<?") == 0) {
      size_t declEnd = content.find("?>", tagStart + 2);
      if (declEnd == std::string::npos) {
        break;
      }
      pos = declEnd + 2;
      continue;
    }
    size_t tagEnd = content.find('>', tagStart + 1);
    if (tagEnd == std::string::npos) {
      break;
    }
    std::string tagText = content.substr(tagStart + 1, tagEnd - tagStart - 1);
    pos = tagEnd + 1;
    if (!tagText.empty() && tagText[0] == '!') {
      continue;
    }

    auto trim = [](std::string s) {
      size_t start = 0;
      while (start < s.size() &&
             std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
      }
      size_t end = s.size();
      while (end > start &&
             std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
      }
      return s.substr(start, end - start);
    };
    std::string trimmed = trim(tagText);
    if (trimmed.empty()) {
      continue;
    }
    bool closing = trimmed[0] == '/';
    bool selfClosing = false;
    size_t len = trimmed.size();
    size_t back = len;
    while (back > 0 &&
           std::isspace(static_cast<unsigned char>(trimmed[back - 1]))) {
      --back;
    }
    if (back > 0 && trimmed[back - 1] == '/') {
      selfClosing = true;
      trimmed = trimmed.substr(0, back - 1);
      trimmed = trim(trimmed);
    }
    size_t nameStart = closing ? 1 : 0;
    while (nameStart < trimmed.size() &&
           std::isspace(static_cast<unsigned char>(trimmed[nameStart]))) {
      ++nameStart;
    }
    size_t nameEnd = nameStart;
    while (nameEnd < trimmed.size() &&
           !std::isspace(static_cast<unsigned char>(trimmed[nameEnd])) &&
           trimmed[nameEnd] != '/') {
      ++nameEnd;
    }
    std::string tagName = trimmed.substr(nameStart, nameEnd - nameStart);
    std::string lowerTag = tagName;
    std::transform(lowerTag.begin(), lowerTag.end(), lowerTag.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    std::string attrBody =
        nameEnd < trimmed.size() ? trimmed.substr(nameEnd) : std::string();

    if (lowerTag == "g") {
      if (closing) {
        if (transformStack.size() > 1) {
          transformStack.pop_back();
        }
      } else {
        SvgTransform combined = transformStack.back();
        std::string transformAttr = getAttributeValue(attrBody, "transform");
        if (!transformAttr.empty()) {
          SvgTransform parsed;
          if (parseTransformList(transformAttr, parsed)) {
            combined = multiplyTransform(combined, parsed);
          }
        }
        transformStack.push_back(combined);
        if (selfClosing && transformStack.size() > 1) {
          transformStack.pop_back();
        }
      }
    } else if (lowerTag == "path" && !closing) {
      std::string d = getAttributeValue(attrBody, "d");
      if (d.empty()) {
        continue;
      }
      std::vector<SvgShape> parsed;
      if (!parseSvgPath(d, parsed)) {
        continue;
      }
      SvgTransform local = transformStack.back();
      std::string transformAttr = getAttributeValue(attrBody, "transform");
      if (!transformAttr.empty()) {
        SvgTransform parsedTransform;
        if (parseTransformList(transformAttr, parsedTransform)) {
          local = multiplyTransform(local, parsedTransform);
        }
      }
      std::string strokeAttr = getAttributeValue(attrBody, "stroke-width");
      for (auto &shape : parsed) {
        applyTransform(local, shape.points);
        if (!strokeAttr.empty()) {
          try {
            float width = std::stof(strokeAttr);
            if (width > 0.0f) {
              shape.strokeWidth = width;
            }
          } catch (...) {
            // ignore malformed stroke widths
          }
        }
        doc.shapes.push_back(std::move(shape));
      }
    }
  }

  normalizeSvgCoordinates(doc);

  // Debug logging
  std::cout << "[SketchProcessor] Loaded SVG: " << doc.shapes.size()
            << " shapes, " << doc.width << "x" << doc.height << std::endl;
  for (size_t i = 0; i < doc.shapes.size(); ++i) {
    std::cout << "  Shape " << i << ": " << doc.shapes[i].points.size()
              << " points, closed=" << doc.shapes[i].closed
              << ", stroke=" << doc.shapes[i].strokeWidth << std::endl;
  }

  return !doc.shapes.empty();
}

#if __has_include(<opencv2/opencv.hpp>)
bool rasterizeSvg(const SvgDocument &doc, cv::Mat &mask) {
  if (doc.width <= 0 || doc.height <= 0) {
    return false;
  }
  mask = cv::Mat::zeros(doc.height, doc.width, CV_8UC1);
  if (mask.empty()) {
    return false;
  }
  for (const auto &shape : doc.shapes) {
    if (shape.points.size() < 2) {
      continue;
    }
    std::vector<cv::Point> contour;
    contour.reserve(shape.points.size());
    for (const auto &pt : shape.points) {
      const int x = static_cast<int>(std::round(
          std::clamp(pt[0], 0.0f, static_cast<float>(doc.width - 1))));
      const int y = static_cast<int>(std::round(
          std::clamp(pt[1], 0.0f, static_cast<float>(doc.height - 1))));
      contour.emplace_back(x, y);
    }
    const int thickness =
        std::max(1, static_cast<int>(std::round(shape.strokeWidth)));
    if (shape.closed) {
      std::vector<std::vector<cv::Point>> polys = {contour};
      cv::fillPoly(mask, polys, cv::Scalar(255));
      cv::polylines(mask, contour, true, cv::Scalar(255), thickness);
    } else {
      cv::polylines(mask, contour, false, cv::Scalar(255), thickness);
    }
  }
  return true;
}

bool loadBitmapMask(const std::string &path, cv::Mat &mask) {
  cv::Mat inputColor = cv::imread(path, cv::IMREAD_COLOR);
  if (inputColor.empty()) {
    return false;
  }

  cv::Mat gray;
  cv::cvtColor(inputColor, gray, cv::COLOR_BGR2GRAY);
  cv::Mat enhanced;
  {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(gray, enhanced);
  }
  cv::Mat blurred;
  cv::GaussianBlur(enhanced, blurred, cv::Size(5, 5), 0.0);

  cv::Mat threshDark;
  cv::Mat threshLight;
  cv::threshold(blurred, threshDark, 0, 255,
                cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
  cv::threshold(blurred, threshLight, 0, 255,
                cv::THRESH_BINARY | cv::THRESH_OTSU);

  const int totalPixels = blurred.rows * blurred.cols;
  const auto foregroundFraction = [&](const cv::Mat &m) -> float {
    return totalPixels > 0
               ? static_cast<float>(cv::countNonZero(m)) / totalPixels
               : 0.0f;
  };

  std::vector<cv::Mat> candidates;
  candidates.push_back(threshDark);
  candidates.push_back(threshLight);

  const int minDim = std::min(blurred.cols, blurred.rows);
  int adaptiveBlock = std::max(15, (minDim / 16) | 1);
  int adaptiveC = 5;
  cv::Mat adaptiveDark;
  cv::Mat adaptiveLight;
  cv::adaptiveThreshold(blurred, adaptiveDark, 255,
                        cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV,
                        adaptiveBlock, adaptiveC);
  cv::adaptiveThreshold(blurred, adaptiveLight, 255,
                        cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY,
                        adaptiveBlock, adaptiveC);
  candidates.push_back(adaptiveDark);
  candidates.push_back(adaptiveLight);

  const float minFrac = 0.01f;
  const float maxFrac = 0.6f;
  auto score = [&](float f) {
    if (f < minFrac || f > maxFrac) {
      return std::numeric_limits<float>::infinity();
    }
    const float mid = 0.2f;
    return std::abs(f - mid);
  };

  float bestScore = std::numeric_limits<float>::infinity();
  size_t bestIdx = 0;
  for (size_t i = 0; i < candidates.size(); ++i) {
    const float fraction = foregroundFraction(candidates[i]);
    const float s = score(fraction);
    if (s < bestScore) {
      bestScore = s;
      bestIdx = i;
    }
  }
  mask = candidates[bestIdx].clone();
  return true;
}
#endif

} // namespace

SketchProcessor::SketchProcessor() = default;

SketchData SketchProcessor::process(const std::string &sketchPath,
                                    float strokeThickness,
                                    float axisOffsetX) const {
  SketchData data;
#if __has_include(<opencv2/opencv.hpp>)
  cv::Mat mask;
  bool isSvg = false;
  SvgDocument svgDoc;

  if (hasSvgExtension(sketchPath)) {
    if (loadSvgDocument(sketchPath, svgDoc)) {
      isSvg = true;
      data.imageWidth = svgDoc.width;
      data.imageHeight = svgDoc.height;
      data.estimatedStroke = strokeThickness > 0.0f ? strokeThickness : 2.0f;

      // Use SVG vector data directly - no rasterization needed!
      data.contours.clear();
      int largestIndex = -1;
      double largestArea = 0.0;

      // Check if SVG uses normalized coordinates (0-1 range) or absolute pixel
      // coordinates Heuristic: if max coordinate is close to 1, it's
      // normalized; if close to width/height, it's absolute
      float maxCoord = 0.0f;
      for (const auto &shape : svgDoc.shapes) {
        for (const auto &pt : shape.points) {
          maxCoord = std::max(maxCoord, std::max(pt[0], pt[1]));
        }
      }

      // If max coordinate is < 2.0, assume normalized (0-1 range)
      // If max coordinate is >= 10, assume absolute pixels
      bool usesNormalizedCoords = (maxCoord < 2.0f);

      std::cout << "[SketchProcessor] SVG max coordinate: " << maxCoord
                << ", treating as: "
                << (usesNormalizedCoords ? "normalized (0-1)"
                                         : "absolute pixels")
                << std::endl;

      for (size_t i = 0; i < svgDoc.shapes.size(); ++i) {
        const auto &shape = svgDoc.shapes[i];
        if (shape.points.size() < 3)
          continue;

        std::vector<std::array<float, 2>> normalized;
        normalized.reserve(shape.points.size());

        for (const auto &pt : shape.points) {
          float nx, ny;
          if (usesNormalizedCoords) {
            // Coordinates are already 0-1, just center them to -0.5 to 0.5
            nx = pt[0] - 0.5f;
            ny = -(pt[1] - 0.5f); // Flip Y
          } else {
            // Coordinates are absolute pixels, normalize them
            nx = pt[0] / static_cast<float>(svgDoc.width) - 0.5f;
            const float nyImage =
                pt[1] / static_cast<float>(svgDoc.height) - 0.5f;
            ny = -nyImage;
          }
          normalized.push_back({nx, ny});
        }

        // Calculate area to find largest shape
        double area = 0.0;
        for (size_t j = 0; j < normalized.size(); ++j) {
          const auto &p1 = normalized[j];
          const auto &p2 = normalized[(j + 1) % normalized.size()];
          area += (p2[0] - p1[0]) * (p2[1] + p1[1]);
        }
        area = std::abs(area * 0.5);

        if (area > largestArea) {
          largestArea = area;
          largestIndex = static_cast<int>(i);
        }

        data.contours.push_back(normalized);
      }

      if (largestIndex >= 0 &&
          largestIndex < static_cast<int>(data.contours.size())) {
        data.contour = data.contours[largestIndex];
      } else if (!data.contours.empty()) {
        data.contour = data.contours.front();
      }

      // Debug: print first few points of largest contour
      if (!data.contour.empty()) {
        std::cout << "[SketchProcessor] SVG main contour ("
                  << data.contour.size() << " points): ";
        size_t numToPrint = std::min(size_t(5), data.contour.size());
        for (size_t i = 0; i < numToPrint; ++i) {
          std::cout << "(" << data.contour[i][0] << "," << data.contour[i][1]
                    << ") ";
        }
        std::cout << std::endl;
      }

      // For heightmap and profile, we still need a rasterized version
      if (!rasterizeSvg(svgDoc, mask)) {
        std::cerr << "[SketchProcessor] WARNING: rasterizeSvg failed!"
                  << std::endl;
      } else {
        std::cout << "[SketchProcessor] Rasterized SVG to " << mask.cols << "x"
                  << mask.rows
                  << ", non-zero pixels: " << cv::countNonZero(mask)
                  << std::endl;
      }
    } else {
      return data;
    }
  }

  if (!isSvg) {
    if (!loadBitmapMask(sketchPath, mask)) {
      return data;
    }
    data.imageWidth = mask.cols;
    data.imageHeight = mask.rows;
    data.estimatedStroke = strokeThickness > 0.0f ? strokeThickness : 2.0f;
  }

  if (!isSvg && mask.empty()) {
    return data;
  }

  // If it's an SVG, and rasterization failed, or if it's a bitmap and mask is
  // empty, return. If it's an SVG, data.imageWidth/Height/estimatedStroke are
  // already set. If it's a bitmap, they are set above. If it's an SVG and
  // rasterization failed, mask will be empty, and we proceed with vector data.
  // If it's an SVG and rasterization succeeded, mask will be populated.
  // If it's a bitmap and loadBitmapMask failed, we return above.
  // If it's a bitmap and loadBitmapMask succeeded, mask will be populated.

  // The following block should only execute if mask is valid (for
  // heightmap/profile) or if it's an SVG and we have vector data, but mask
  // might be empty. The original code had
  // data.imageWidth/Height/estimatedStroke set here, but for SVG, they are set
  // earlier. For bitmap, they are set in the !isSvg block. So, these lines are
  // removed from here.

  if (!isSvg && !mask.empty()) {
    if (strokeThickness > 0.0f) {
      int radius =
          std::clamp(static_cast<int>(std::round(strokeThickness)), 1, 10);
      const int kSize = 1 + 2 * radius;
      cv::Mat dilateKernel =
          cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kSize, kSize));
      cv::dilate(mask, mask, dilateKernel);
    }

    cv::Mat adaptiveKernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::Mat closed;
    cv::morphologyEx(mask, closed, cv::MORPH_CLOSE, adaptiveKernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(closed, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    cv::Mat filled;
    int largestIndex = -1;
    double largestArea = 0.0;
    if (!contours.empty()) {
      filled = cv::Mat::zeros(closed.size(), CV_8UC1);
      cv::drawContours(filled, contours, -1, cv::Scalar(255), cv::FILLED);
      for (size_t i = 0; i < contours.size(); ++i) {
        const double area = std::abs(cv::contourArea(contours[i]));
        if (area > largestArea) {
          largestArea = area;
          largestIndex = static_cast<int>(i);
        }
      }
      data.contours.clear();
      const double areaThreshold = std::max(100.0, largestArea * 0.05);
      for (size_t i = 0; i < contours.size(); ++i) {
        const double area = std::abs(cv::contourArea(contours[i]));
        if (area < areaThreshold) {
          continue;
        }
        cv::Mat componentMask = cv::Mat::zeros(closed.size(), CV_8UC1);
        std::vector<std::vector<cv::Point>> single = {contours[i]};
        cv::drawContours(componentMask, single, -1, cv::Scalar(255),
                         cv::FILLED);
        std::vector<std::vector<cv::Point>> componentContours;
        cv::findContours(componentMask, componentContours, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_SIMPLE);
        if (componentContours.empty()) {
          continue;
        }
        const auto &poly = componentContours.front();
        std::vector<cv::Point> simplified;
        const double epsilon = 0.003 * cv::arcLength(poly, true);
        cv::approxPolyDP(poly, simplified, epsilon, true);
        const auto &finalContour = simplified.empty() ? poly : simplified;
        std::vector<std::array<float, 2>> normalized;
        normalized.reserve(finalContour.size());
        for (const auto &pt : finalContour) {
          const float nx =
              static_cast<float>(pt.x) / static_cast<float>(closed.cols) - 0.5f;
          const float nyImage =
              static_cast<float>(pt.y) / static_cast<float>(closed.rows) - 0.5f;
          const float ny = -nyImage;
          normalized.push_back({nx, ny});
        }
        if (!normalized.empty()) {
          data.contours.push_back(normalized);
          if (static_cast<int>(i) == largestIndex || data.contour.empty()) {
            data.contour = normalized;
          }
        }
      }
      if (data.contour.empty() && !data.contours.empty()) {
        data.contour = data.contours.front();
      }

      // Use filled for profile/heightmap
      mask = filled;
    }
  }

  // Build profile and heightmap from mask (works for both SVG and bitmap)
  if (!mask.empty()) {
    const int samples = 64;
    const float axisNormalized = std::clamp(0.5f + axisOffsetX, 0.0f, 1.0f);
    const float axisPixel =
        axisNormalized * static_cast<float>(std::max(mask.cols - 1, 0));
    const int axisCol = std::clamp(
        static_cast<int>(std::round(axisPixel)), 0, std::max(mask.cols - 1, 0));
    for (int i = 0; i < samples; ++i) {
      const float fy = static_cast<float>(i) / static_cast<float>(samples - 1);
      const int row =
          std::clamp(static_cast<int>(fy * (mask.rows - 1)), 0, mask.rows - 1);
      const uchar *ptr = mask.ptr<uchar>(row);
      int leftmost = axisCol;
      bool found = false;
      for (int col = 0; col <= axisCol; ++col) {
        if (ptr[col] > 0) { // take the furthest point on the left half
          leftmost = col;
          found = true;
          break;
        }
      }
      if (found) {
        const float radiusPixels = axisPixel - static_cast<float>(leftmost);
        if (radiusPixels <= 0.0f) {
          continue;
        }
        const float radius = radiusPixels / static_cast<float>(mask.cols);
        const float yImage =
            static_cast<float>(row) / static_cast<float>(mask.rows) - 0.5f;
        const float y = -yImage;
        data.profile.push_back({radius, y});
      }
    }
  }

  const unsigned int res = 64;
  data.heightmap.assign(res * res, 0.0f);
  if (!mask.empty()) {
    cv::Mat resized;
    cv::resize(mask, resized, cv::Size(res, res), 0, 0, cv::INTER_AREA);
    for (unsigned int y = 0; y < res; ++y) {
      for (unsigned int x = 0; x < res; ++x) {
        const float v = static_cast<float>(resized.at<uchar>(y, x)) / 255.0f;
        data.heightmap[y * res + x] = v;
      }
    }
  } else {
    const float sigma = 0.25f;
    for (unsigned int y = 0; y < res; ++y) {
      for (unsigned int x = 0; x < res; ++x) {
        const float fx =
            static_cast<float>(x) / static_cast<float>(res - 1) - 0.5f;
        const float fy =
            static_cast<float>(y) / static_cast<float>(res - 1) - 0.5f;
        data.heightmap[y * res + x] =
            std::exp(-(fx * fx + fy * fy) / (2.0f * sigma * sigma));
      }
    }
  }
#else
  data.contour = {{-0.3f, -0.3f}, {0.3f, -0.3f}, {0.3f, 0.3f}, {-0.3f, 0.3f}};
  data.contours = {data.contour};
  data.imageWidth = 576;
  data.imageHeight = 640;
  data.estimatedStroke = strokeThickness > 0.0f ? strokeThickness : 2.0f;
  const unsigned int res = 64;
  data.heightmap.resize(res * res);
  const float sigma = 0.25f;
  for (unsigned int y = 0; y < res; ++y) {
    for (unsigned int x = 0; x < res; ++x) {
      const float fx =
          static_cast<float>(x) / static_cast<float>(res - 1) - 0.5f;
      const float fy =
          static_cast<float>(y) / static_cast<float>(res - 1) - 0.5f;
      data.heightmap[y * res + x] =
          std::exp(-(fx * fx + fy * fy) / (2.0f * sigma * sigma));
    }
  }
#endif

  if (data.profile.empty()) {
    data.profile = {{0.0f, -0.4f}, {0.3f, 0.0f}, {0.1f, 0.4f}};
  }
  return data;
}

std::string SketchProcessor::exportToSvg(const SketchData &data,
                                         int widthOverride,
                                         int heightOverride) const {
  if (data.contours.empty()) {
    return {};
  }
  const int width = widthOverride > 0
                        ? widthOverride
                        : (data.imageWidth > 0 ? data.imageWidth : 576);
  const int height = heightOverride > 0
                         ? heightOverride
                         : (data.imageHeight > 0 ? data.imageHeight : 640);
  nlohmann::json meta;
  meta["width"] = width;
  meta["height"] = height;
  meta["shapes"] = nlohmann::json::array();
  for (const auto &loop : data.contours) {
    if (loop.size() < 2) {
      continue;
    }
    nlohmann::json shape;
    shape["closed"] = true;
    shape["strokeWidth"] =
        data.estimatedStroke > 0.0f ? data.estimatedStroke : 2.0f;
    nlohmann::json points = nlohmann::json::array();
    for (const auto &pt : loop) {
      const float x = (pt[0] + 0.5f) * static_cast<float>(width);
      const float y = (0.5f - pt[1]) * static_cast<float>(height);
      points.push_back({x, y});
    }
    shape["points"] = points;
    meta["shapes"].push_back(shape);
  }
  if (meta["shapes"].empty()) {
    return {};
  }
  auto escapeXml = [](const std::string &value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char c : value) {
      switch (c) {
      case '&':
        escaped += "&amp;";
        break;
      case '<':
        escaped += "&lt;";
        break;
      case '>':
        escaped += "&gt;";
        break;
      default:
        escaped += c;
        break;
      }
    }
    return escaped;
  };

  auto buildPath = [&](const std::vector<std::array<float, 2>> &loop) {
    if (loop.empty()) {
      return std::string();
    }
    std::ostringstream ss;
    const auto toPoint = [&](const std::array<float, 2> &p) {
      const float x = (p[0] + 0.5f) * static_cast<float>(width);
      const float y = (0.5f - p[1]) * static_cast<float>(height);
      return std::pair<float, float>(x, y);
    };
    const auto [x0, y0] = toPoint(loop.front());
    ss << "M " << x0 << " " << y0;
    for (size_t i = 1; i < loop.size(); ++i) {
      const auto [x, y] = toPoint(loop[i]);
      ss << " L " << x << " " << y;
    }
    ss << " Z";
    return ss.str();
  };

  std::ostringstream svg;
  svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
      << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " "
      << height << "\">"
      << "<metadata id=\"sketch-data\">" << escapeXml(meta.dump())
      << "</metadata><g fill=\"none\" stroke=\"#000\" stroke-linecap=\"round\" "
      << "stroke-linejoin=\"round\">";
  const float stroke =
      data.estimatedStroke > 0.0f ? data.estimatedStroke : 2.0f;
  for (const auto &loop : data.contours) {
    const std::string d = buildPath(loop);
    if (d.empty()) {
      continue;
    }
    svg << "<path d=\"" << d << "\" stroke-width=\"" << stroke << "\" />";
  }
  svg << "</g></svg>";
  return svg.str();
}

std::string
SketchProcessor::vectorizeBitmapWithPotrace(const std::string &imagePath,
                                            float strokeThickness) const {
#if __has_include(<opencv2/opencv.hpp>)
  cv::Mat mask;
  if (!loadBitmapMask(imagePath, mask)) {
    return {};
  }
  if (mask.empty()) {
    return {};
  }
  if (strokeThickness > 0.0f) {
    int radius =
        std::clamp(static_cast<int>(std::round(strokeThickness)), 1, 10);
    const int kSize = 1 + 2 * radius;
    cv::Mat dilateKernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kSize, kSize));
    cv::dilate(mask, mask, dilateKernel);
  }
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

  const auto tempDir = std::filesystem::temp_directory_path();
  const auto timestamp =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  const std::filesystem::path pbmPath =
      tempDir / ("sketch3d_" + std::to_string(timestamp) + ".pbm");
  const std::filesystem::path svgPath =
      tempDir / ("sketch3d_" + std::to_string(timestamp) + ".svg");

  std::ofstream pbm(pbmPath);
  if (!pbm.is_open()) {
    return {};
  }
  pbm << "P1\n" << mask.cols << " " << mask.rows << "\n";
  for (int y = 0; y < mask.rows; ++y) {
    const uchar *row = mask.ptr<uchar>(y);
    for (int x = 0; x < mask.cols; ++x) {
      pbm << (row[x] > 0 ? '1' : '0') << ' ';
    }
    pbm << '\n';
  }
  pbm.close();

  std::string command =
      "potrace -s -o " + svgPath.string() + " " + pbmPath.string();
  int status = std::system(command.c_str());
  std::error_code ec;
  std::filesystem::remove(pbmPath, ec);
  if (status != 0) {
    std::filesystem::remove(svgPath, ec);
    return {};
  }

  std::ifstream svgFile(svgPath);
  if (!svgFile.is_open()) {
    std::filesystem::remove(svgPath, ec);
    return {};
  }
  std::stringstream buffer;
  buffer << svgFile.rdbuf();
  svgFile.close();
  std::filesystem::remove(svgPath, ec);
  return buffer.str();
#else
  (void)imagePath;
  (void)strokeThickness;
  return {};
#endif
}
