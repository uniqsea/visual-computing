#include "api/HttpServer.h"
#include "sketch/SketchProcessor.h"
#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <netinet/in.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace {
constexpr const char *kAssetsPrefix = "/api/assets/";
}

HttpServer::HttpServer(PipelineController &controller)
    : pipeline(controller), uploadDir("data/uploads") {
  std::filesystem::create_directories(uploadDir);
}

void HttpServer::start(int port) {
  int serverFd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (serverFd < 0) {
    throw std::runtime_error("Failed to create socket");
  }

  int opt = 1;
  setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(port);

  if (bind(serverFd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) <
      0) {
    throw std::runtime_error("Failed to bind socket");
  }

  if (listen(serverFd, 16) < 0) {
    throw std::runtime_error("Failed to listen on socket");
  }

  std::cout << "[HttpServer] Listening on port " << port << std::endl;
  listenLoop(serverFd);
}

void HttpServer::listenLoop(int serverFd) {
  while (true) {
    sockaddr_in clientAddr{};
    socklen_t clientLen = sizeof(clientAddr);
    int clientFd =
        accept(serverFd, reinterpret_cast<sockaddr *>(&clientAddr), &clientLen);
    if (clientFd < 0) {
      continue;
    }
    std::thread(&HttpServer::handleClient, this, clientFd).detach();
  }
}

void HttpServer::handleClient(int clientFd) {
  std::string request;
  char buffer[65536]; // Increased from 4096 to handle large base64 payloads
  size_t headerEnd = std::string::npos;
  size_t contentLength = 0;

  while (true) {
    ssize_t bytes = recv(clientFd, buffer, sizeof(buffer), 0);
    if (bytes <= 0) {
      break;
    }
    request.append(buffer, bytes);
    headerEnd = request.find("\r\n\r\n");
    if (headerEnd != std::string::npos) {
      std::string header = request.substr(0, headerEnd);
      std::string contentLenStr = getHeaderValue(header, "Content-Length");
      if (!contentLenStr.empty()) {
        try {
          contentLength = static_cast<size_t>(std::stoul(contentLenStr));
        } catch (...) {
          contentLength = 0;
        }
      }
      break;
    }
  }

  if (request.empty()) {
    close(clientFd);
    return;
  }

  if (headerEnd == std::string::npos) {
    sendServerError(clientFd, "Malformed request");
    close(clientFd);
    return;
  }
  const size_t bodyStart = headerEnd + 4;
  std::string headerText = request.substr(0, headerEnd);
  std::string body = request.substr(bodyStart);
  while (body.size() < contentLength) {
    ssize_t bytes = recv(clientFd, buffer, sizeof(buffer), 0);
    if (bytes <= 0) {
      break;
    }
    body.append(buffer, bytes);
  }

  // Verify we received the complete body
  if (contentLength > 0) {
    if (body.size() != contentLength) {
      std::cerr << "[HttpServer] Body size mismatch: expected " << contentLength
                << " bytes, got " << body.size() << " bytes\n";
      sendServerError(clientFd, "Incomplete request body");
      close(clientFd);
      return;
    }

    // Debug logging to check for truncation
    if (body.size() > 20) {
      std::string tail = body.substr(body.size() - 20);
      std::cout << "[HttpServer] Body received (" << body.size()
                << " bytes). Tail: " << tail << std::endl;
    }
  }

  std::istringstream headerStream(headerText);
  std::string requestLine;
  std::getline(headerStream, requestLine);
  if (!requestLine.empty() && requestLine.back() == '\r') {
    requestLine.pop_back();
  }
  std::istringstream lineStream(requestLine);
  std::string method;
  std::string path;
  std::string version;
  lineStream >> method >> path >> version;
  const auto queryPos = path.find('?');
  if (queryPos != std::string::npos) {
    path = path.substr(0, queryPos);
  }

  if (method == "POST" && path == "/api/sketch") {
    handleSketchRequest(body, clientFd);
  } else if (method == "POST" && path == "/api/vectorize") {
    handleVectorizeRequest(body, clientFd);
  } else if (method == "GET" && path == "/api/result/latest") {
    handleLatestRequest(clientFd);
  } else if (method == "GET" && path.rfind(kAssetsPrefix, 0) == 0) {
    handleAssetRequest(path.substr(std::strlen(kAssetsPrefix)), clientFd);
  } else {
    sendNotFound(clientFd);
  }
  close(clientFd);
}

void HttpServer::handleVectorizeRequest(const std::string &body, int clientFd) {
  std::string data;
  float strokeThickness = 0.0f;
  if (!parseVectorizePayload(body, data, strokeThickness)) {
    sendServerError(clientFd, "Invalid JSON payload");
    return;
  }

  auto decoded = decodeBase64(data);
  if (decoded.empty()) {
    sendServerError(clientFd, "Invalid image data");
    return;
  }

  const std::string defaultName = "vectorize-input";
  std::string savedPath =
      saveUploadedFile(defaultName + std::string(".png"), decoded);
  const std::filesystem::path savedPathFs(savedPath);
  const std::string ext = savedPathFs.extension().string();
  auto cleanup = [&]() {
    std::error_code ec;
    std::filesystem::remove(savedPathFs, ec);
  };

  std::string svg;
  SketchProcessor processor;
  if (!ext.empty() && (ext == ".svg" || ext == ".SVG")) {
    std::ifstream in(savedPath);
    if (!in.is_open()) {
      cleanup();
      sendServerError(clientFd, "Failed to read SVG data.");
      return;
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    svg = buffer.str();
    cleanup();
  } else {
    svg = processor.vectorizeBitmapWithPotrace(savedPath, strokeThickness);
    cleanup();
    if (svg.empty()) {
      sendServerError(
          clientFd,
          "Vectorization failed to detect sufficient foreground strokes.");
      return;
    }
  }

  nlohmann::json response = {{"svg", svg}};
  sendResponse(clientFd, 200, "OK", "application/json", response.dump());
}

void HttpServer::handleSketchRequest(const std::string &body, int clientFd) {
  std::string mode;
  std::string filename;
  std::string data;
  SketchRequest request;
  if (!parseSketchPayload(body, mode, filename, data, request)) {
    std::cerr << "[HttpServer] Failed to parse /api/sketch payload\n";
    sendServerError(clientFd, "Invalid JSON payload");
    return;
  }

  auto decoded = decodeBase64(data);
  if (decoded.empty()) {
    std::cerr << "[HttpServer] Base64 decode failed for uploaded sketch\n";
    sendServerError(clientFd, "Invalid image data");
    return;
  }

  std::string savedPath = saveUploadedFile(filename, decoded);
  request.mode = mode;
  request.sketchPath = savedPath;

  try {
    RenderResult result = pipeline.handleRequest(request);
    {
      std::lock_guard<std::mutex> lock(resultMutex);
      lastResult = result;
    }
    const std::filesystem::path imageName =
        std::filesystem::path(result.renderImagePath).filename();
    const std::filesystem::path meshName =
        std::filesystem::path(result.meshJsonPath).filename();
    nlohmann::json response = {
        {"status", "ok"},
        {"token", result.relativeDir},
        {"image", makeHttpPath(result.relativeDir, imageName.string())},
        {"mesh", makeHttpPath(result.relativeDir, meshName.string())}};
    sendResponse(clientFd, 200, "OK", "application/json", response.dump());
    std::cout << "[HttpServer] Processed sketch (" << mode << ") -> "
              << result.relativeDir << std::endl;
  } catch (const std::exception &ex) {
    std::cerr << "[HttpServer] Exception during processing: " << ex.what()
              << std::endl;
    sendServerError(clientFd, "Processing error");
  }
}

void HttpServer::handleLatestRequest(int clientFd) {
  std::optional<RenderResult> copy;
  {
    std::lock_guard<std::mutex> lock(resultMutex);
    copy = lastResult;
  }
  if (!copy.has_value()) {
    sendNotFound(clientFd);
    return;
  }
  const std::filesystem::path imageName =
      std::filesystem::path(copy->renderImagePath).filename();
  const std::filesystem::path meshName =
      std::filesystem::path(copy->meshJsonPath).filename();
  const std::string imagePath =
      makeHttpPath(copy->relativeDir, imageName.string());
  const std::string meshPath =
      makeHttpPath(copy->relativeDir, meshName.string());
  std::ostringstream body;
  body << "{\"image\":\"" << imagePath << "\",\"mesh\":\"" << meshPath << "\"}";
  sendResponse(clientFd, 200, "OK", "application/json", body.str());
}

void HttpServer::handleAssetRequest(const std::string &relativePath,
                                    int clientFd) {
  std::string cleanPath = relativePath;
  while (!cleanPath.empty() && cleanPath.front() == '/') {
    cleanPath.erase(cleanPath.begin());
  }
  std::filesystem::path base = pipeline.getOutputDirectory();
  std::filesystem::path requested = std::filesystem::path(cleanPath);
  std::filesystem::path canonicalBase;
  std::filesystem::path canonicalTarget;
  try {
    canonicalBase = std::filesystem::weakly_canonical(base);
    canonicalTarget = std::filesystem::weakly_canonical(base / requested);
  } catch (...) {
    sendNotFound(clientFd);
    return;
  }
  const std::string baseStr = canonicalBase.string();
  const std::string targetStr = canonicalTarget.string();
  if (targetStr.rfind(baseStr, 0) != 0 ||
      !std::filesystem::is_regular_file(canonicalTarget)) {
    sendNotFound(clientFd);
    return;
  }
  std::ifstream file(canonicalTarget, std::ios::binary);
  if (!file) {
    sendNotFound(clientFd);
    return;
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  sendResponse(clientFd, 200, "OK", guessContentType(canonicalTarget.string()),
               buffer.str());
}

bool HttpServer::parseSketchPayload(const std::string &body, std::string &mode,
                                    std::string &filename, std::string &data,
                                    SketchRequest &request) const {
  try {
    auto json = nlohmann::json::parse(body);
    if (!json.contains("data") || json["data"].is_null() ||
        !json["data"].is_string()) {
      std::cerr << "[HttpServer] JSON payload missing data field\n";
      std::cerr << "[HttpServer] Raw payload (truncated): "
                << body.substr(0, 256) << (body.size() > 256 ? "..." : "")
                << std::endl;
      return false;
    }
    data = json["data"].get<std::string>();
    if (data.empty()) {
      std::cerr << "[HttpServer] JSON data field is empty\n";
      return false;
    }
    mode = json.value("mode", "extrusion");
    filename = json.value("filename", "sketch.png");
    if (json.contains("settings") && json["settings"].is_object()) {
      const auto &settings = json["settings"];
      request.sketchThickness =
          settings.value("sketchThickness", request.sketchThickness);
      request.extrusionDepth =
          settings.value("extrusionDepth", request.extrusionDepth);
      request.extrusionSmoothSteps =
          settings.value("extrusionSmoothSteps", request.extrusionSmoothSteps);

      request.revolutionSegments =
          settings.value("revolutionSegments", request.revolutionSegments);
      bool capEnds =
          settings.value("revolutionCapEnds", false);
      request.revolutionCapBottom =
          settings.value("revolutionCapBottom",
                         settings.value("revolutionCapEnds",
                                        request.revolutionCapBottom ? true : capEnds));
      request.revolutionCapTop =
          settings.value("revolutionCapTop",
                         settings.value("revolutionCapEnds",
                                        request.revolutionCapTop ? true : capEnds));
      request.revolutionAxisOffsetX = settings.value(
          "revolutionAxisOffsetX", request.revolutionAxisOffsetX);
      request.revolutionHollow =
          settings.value("revolutionHollow", request.revolutionHollow);
      request.revolutionWallThickness =
          settings.value("revolutionWallThickness",
                         request.revolutionWallThickness);
      request.revolutionAngleDegrees =
          settings.value("revolutionAngleDegrees",
                         request.revolutionAngleDegrees);

      request.heightScale = settings.value("heightScale", request.heightScale);
      request.heightWithBase =
          settings.value("heightWithBase", request.heightWithBase);
      request.heightBlurSigma =
          settings.value("heightBlurSigma", request.heightBlurSigma);
      request.heightResolution = settings.value(
          "heightResolution", request.heightResolution);
      request.heightBulgeStrength =
          settings.value("heightBulgeStrength", request.heightBulgeStrength);
    }
    return true;
  } catch (const nlohmann::json::exception &ex) {
    std::cerr << "[HttpServer] JSON parse error: " << ex.what() << "\n";
    std::cerr << "[HttpServer] Raw payload (truncated): " << body.substr(0, 256)
              << (body.size() > 256 ? "..." : "") << std::endl;
    return false;
  }
}

bool HttpServer::parseVectorizePayload(const std::string &body,
                                       std::string &data,
                                       float &strokeThickness) const {
  try {
    auto json = nlohmann::json::parse(body);
    data = json.value("data", "");
    strokeThickness = json.value("strokeThickness", 0.0f);
    return !data.empty();
  } catch (const nlohmann::json::exception &ex) {
    std::cerr << "[HttpServer] Vectorize JSON parse error: " << ex.what()
              << "\n";
    return false;
  }
}

std::vector<unsigned char> HttpServer::decodeBase64(const std::string &input) {
  static constexpr unsigned char kInvalid = 255;
  static unsigned char table[256];
  static std::once_flag flag;
  std::call_once(flag, []() {
    std::fill(std::begin(table), std::end(table), kInvalid);
    const std::string chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (unsigned int i = 0; i < chars.size(); ++i) {
      table[static_cast<unsigned char>(chars[i])] =
          static_cast<unsigned char>(i);
    }
  });

  std::vector<unsigned char> output;
  unsigned int val = 0;
  int valb = -8;
  for (unsigned char c : input) {
    if (table[c] == kInvalid) {
      if (c == '=') {
        break;
      }
      continue;
    }
    val = (val << 6) + table[c];
    valb += 6;
    if (valb >= 0) {
      output.push_back(static_cast<unsigned char>((val >> valb) & 0xFF));
      valb -= 8;
    }
  }
  return output;
}

std::string
HttpServer::saveUploadedFile(const std::string &filename,
                             const std::vector<unsigned char> &data) const {
  std::filesystem::create_directories(uploadDir);
  std::filesystem::path sanitized = filename;
  sanitized = sanitized.filename();
  if (sanitized.empty() || sanitized.string() == ".") {
    sanitized = "sketch.png";
  }
  const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  std::stringstream ss;
  ss << now << "_" << sanitized.string();
  std::filesystem::path filePath = std::filesystem::path(uploadDir) / ss.str();
  std::ofstream file(filePath, std::ios::binary);
  file.write(reinterpret_cast<const char *>(data.data()), data.size());
  return filePath.string();
}

std::string HttpServer::makeHttpPath(const std::string &token,
                                     const std::string &file) const {
  std::ostringstream oss;
  oss << "/api/assets/" << token << "/" << file;
  return oss.str();
}

void HttpServer::sendResponse(int clientFd, int status,
                              const std::string &statusText,
                              const std::string &contentType,
                              const std::string &body) {
  std::ostringstream response;
  response << "HTTP/1.1 " << status << " " << statusText << "\r\n";
  response << "Content-Type: " << contentType << "\r\n";
  response << "Content-Length: " << body.size() << "\r\n";
  response << "Connection: close\r\n\r\n";
  response << body;
  const std::string data = response.str();
  send(clientFd, data.c_str(), data.size(), 0);
}

void HttpServer::sendNotFound(int clientFd) {
  const std::string body = "{\"error\":\"Not Found\"}";
  sendResponse(clientFd, 404, "Not Found", "application/json", body);
}

void HttpServer::sendServerError(int clientFd, const std::string &message) {
  std::string body = "{\"error\":\"" + message + "\"}";
  std::cerr << "[HttpServer] 500 Internal Server Error: " << message
            << std::endl;
  sendResponse(clientFd, 500, "Internal Server Error", "application/json",
               body);
}

std::string HttpServer::guessContentType(const std::string &path) {
  if (path.rfind(".png") != std::string::npos) {
    return "image/png";
  }
  if (path.rfind(".bmp") != std::string::npos) {
    return "image/bmp";
  }
  if (path.rfind(".json") != std::string::npos) {
    return "application/json";
  }
  return "application/octet-stream";
}

std::string HttpServer::getHeaderValue(const std::string &headers,
                                       const std::string &key) {
  std::istringstream stream(headers);
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    const auto pos = line.find(':');
    if (pos == std::string::npos) {
      continue;
    }
    std::string name = line.substr(0, pos);
    if (name == key) {
      std::string value = line.substr(pos + 1);
      size_t start = value.find_first_not_of(" \t");
      size_t end = value.find_last_not_of(" \t");
      if (start == std::string::npos) {
        return {};
      }
      return value.substr(start, end - start + 1);
    }
  }
  return {};
}
