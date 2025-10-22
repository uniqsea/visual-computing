#include "Application.h"
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "==================================" << std::endl;
    std::cout << "Real-time Video Processing" << std::endl;
    std::cout << "Visual Computing Assignment 2" << std::endl;
    std::cout << "==================================" << std::endl;
    
    Application app;
    
    if (!app.initialize(1280, 720, "Real-time Video Processing")) {
        std::cerr << "Failed to initialize application" << std::endl;
        return -1;
    }
    
    app.run();
    app.shutdown();
    
    std::cout << "Application exited successfully" << std::endl;
    return 0;
}

