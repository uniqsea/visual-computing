#include "sketch/SketchProcessor.h"
#if __has_include(<opencv2/opencv.hpp>)
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif
#include <algorithm>
#include <array>
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

bool parseSvgPath(const std::string &d, SvgShape &shape) {
  std::stringstream ss(d);
  char cmd;
  float args[6];
  std::array<float, 2> current = {0.0f, 0.0f};
  std::array<float, 2> start = {0.0f, 0.0f};

  // Simple parser for M, L, C, Z (absolute) and m, l, c, z (relative)
  // This is not a full SVG parser but covers Potrace output and basic shapes.
  while (ss >> cmd) {
    if (cmd == 'M' || cmd == 'm') {
      if (!(ss >> args[0] >> args[1]))
        break;
      if (cmd == 'm') {
        args[0] += current[0];
        args[1] += current[1];
      }
      current = {args[0], args[1]};
      start = current;
      shape.points.push_back(current);
    } else if (cmd == 'L' || cmd == 'l') {
      if (!(ss >> args[0] >> args[1]))
        break;
      if (cmd == 'l') {
        args[0] += current[0];
        args[1] += current[1];
      }
      current = {args[0], args[1]};
      shape.points.push_back(current);
    } else if (cmd == 'C' || cmd == 'c') {
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
        shape.points.push_back({x, y});
      }
      current = {args[4], args[5]};
    } else if (cmd == 'Z' || cmd == 'z') {
      shape.closed = true;
      // Close loop if needed
      if (std::abs(current[0] - start[0]) > 0.1f ||
          std::abs(current[1] - start[1]) > 0.1f) {
        shape.points.push_back(start);
      }
      current = start;
    } else {
      // Skip unknown or comma
      if (cmd != ',') {
        // Try to read as number if implicit command (e.g. L x y x y)
        // For simplicity, we assume explicit commands for now, or just skip.
      }
    }
  }
  return !shape.points.empty();
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

  // 2. Fallback: Parse standard SVG paths
  // Extract width/height
  auto findAttr = [&](const std::string &s,
                      const std::string &attr) -> std::string {
    auto pos = s.find(attr + "=\"");
    if (pos == std::string::npos)
      return "";
    pos += attr.size() + 2;
    auto end = s.find("\"", pos);
    if (end == std::string::npos)
      return "";
    return s.substr(pos, end - pos);
  };

  std::string wStr = findAttr(content, "width");
  std::string hStr = findAttr(content, "height");
  if (!wStr.empty())
    doc.width = std::atoi(wStr.c_str());
  if (!hStr.empty())
    doc.height = std::atoi(hStr.c_str());

  // Find all <path> tags
  doc.shapes.clear();
  size_t pos = 0;
  while ((pos = content.find("<path", pos)) != std::string::npos) {
    size_t end = content.find("/>", pos);
    size_t end2 = content.find("</path>", pos);
    if (end == std::string::npos)
      end = end2;
    if (end == std::string::npos)
      break;

    std::string tag = content.substr(pos, end - pos);
    std::string d = findAttr(tag, "d");
    if (!d.empty()) {
      SvgShape s;
      s.strokeWidth = 2.0f; // Default
      if (parseSvgPath(d, s)) {
        doc.shapes.push_back(std::move(s));
      }
    }
    pos = end;
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
                                    float strokeThickness) const {
  SketchData data;
#if __has_include(<opencv2/opencv.hpp>)
  cv::Mat mask;
  if (hasSvgExtension(sketchPath)) {
    SvgDocument doc;
    if (!loadSvgDocument(sketchPath, doc) || !rasterizeSvg(doc, mask)) {
      return data;
    }
  } else {
    if (!loadBitmapMask(sketchPath, mask)) {
      return data;
    }
  }

  if (mask.empty()) {
    return data;
  }
  data.imageWidth = mask.cols;
  data.imageHeight = mask.rows;
  data.estimatedStroke = strokeThickness > 0.0f ? strokeThickness : 2.0f;

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
      cv::drawContours(componentMask, single, -1, cv::Scalar(255), cv::FILLED);
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
  }

  if (!filled.empty()) {
    const int samples = 64;
    const float center = 0.5f * static_cast<float>(filled.cols - 1);
    for (int i = 0; i < samples; ++i) {
      const float fy = static_cast<float>(i) / static_cast<float>(samples - 1);
      const int row = std::clamp(static_cast<int>(fy * (filled.rows - 1)), 0,
                                 filled.rows - 1);
      const uchar *ptr = filled.ptr<uchar>(row);
      float maxDist = 0.0f;
      for (int col = 0; col < filled.cols; ++col) {
        if (ptr[col] > 0) {
          const float dist = static_cast<float>(col) - center;
          if (dist > maxDist) {
            maxDist = dist;
          }
        }
      }
      if (maxDist > 0.0f) {
        const float radius = maxDist / static_cast<float>(filled.cols);
        const float yImage =
            static_cast<float>(row) / static_cast<float>(filled.rows) - 0.5f;
        const float y = -yImage;
        data.profile.push_back({radius, y});
      }
    }
  }

  const unsigned int res = 64;
  data.heightmap.assign(res * res, 0.0f);
  if (!filled.empty()) {
    cv::Mat resized;
    cv::resize(filled, resized, cv::Size(res, res), 0, 0, cv::INTER_AREA);
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
