#pragma once

#include "core/PipelineController.h"
#include <mutex>
#include <optional>
#include <string>
#include <vector>

class HttpServer {
public:
  explicit HttpServer(PipelineController &controller);
  void start(int port);

private:
  PipelineController &pipeline;
  std::optional<RenderResult> lastResult;
  std::mutex resultMutex;
  std::string uploadDir;

  void listenLoop(int serverFd);
  void handleClient(int clientFd);
  void handleSketchRequest(const std::string &body, int clientFd);
  void handleVectorizeRequest(const std::string &body, int clientFd);
  void handleLatestRequest(int clientFd);
  void handleAssetRequest(const std::string &path, int clientFd);

  bool parseSketchPayload(const std::string &body, std::string &mode,
                          std::string &filename, std::string &data,
                          SketchRequest &request) const;
  bool parseVectorizePayload(const std::string &body, std::string &data,
                             float &strokeThickness) const;
  static std::vector<unsigned char> decodeBase64(const std::string &input);
  std::string saveUploadedFile(const std::string &filename,
                               const std::vector<unsigned char> &data) const;
  std::string makeHttpPath(const std::string &token,
                           const std::string &file) const;
  static std::string getHeaderValue(const std::string &headers,
                                    const std::string &key);

  static void sendResponse(int clientFd, int status,
                           const std::string &statusText,
                           const std::string &contentType,
                           const std::string &body);
  static void sendNotFound(int clientFd);
  static void sendServerError(int clientFd, const std::string &message);
  static std::string guessContentType(const std::string &path);
};
