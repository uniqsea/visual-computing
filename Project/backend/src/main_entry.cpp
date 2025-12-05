#include "core/PipelineController.h"
#include "api/HttpServer.h"
#include <iostream>

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  try {
    PipelineController controller;
    HttpServer server(controller);
    server.start(8080);
  } catch (const std::exception &ex) {
    std::cerr << "Fatal error: " << ex.what() << std::endl;
    return 1;
  }
  return 0;
}
