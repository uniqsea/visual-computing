#pragma once

#include <GLFW/glfw3.h>
#include <memory>
#include "VideoCapture.h"
#include "Renderer.h"
#include "ShaderProgram.h"
#include "Texture.h"
#include "GUI.h"
#include "filters/FilterManager.h"
#include "transforms/AffineTransform.h"
#include "utils/Timer.h"
#include "utils/PerformanceLogger.h"

class Application {
public:
    Application();
    ~Application();
    
    bool initialize(int width, int height, const char* title);
    void run();
    void shutdown();
    
private:
    // Window
    GLFWwindow* window;
    int windowWidth;
    int windowHeight;
    
    // Core components
    std::unique_ptr<VideoCapture> videoCapture;
    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<Texture> videoTexture;
    std::unique_ptr<GUI> gui;
    std::unique_ptr<FilterManager> filterManager;
    std::unique_ptr<AffineTransform> transform;
    std::unique_ptr<Timer> timer;
    std::unique_ptr<PerformanceLogger> perfLogger;
    
    // Shaders
    std::unique_ptr<ShaderProgram> basicShader;
    std::unique_ptr<ShaderProgram> pixelationShader;
    std::unique_ptr<ShaderProgram> cartoonShader;
    
    // State
    bool isRunning;
    cv::Mat currentFrame;
    
    // Mouse state for transform interaction
    bool leftMousePressed;
    bool rightMousePressed;
    double lastMouseX;
    double lastMouseY;
    
    // Private methods
    bool initializeGLFW();
    bool initializeShaders();
    bool initializeVideoCapture();
    
    void processFrame();
    void renderFrame();
    void updateGUI();
    void handleInput();
    
    void applyFilter(cv::Mat& frame);
    void applyTransform(cv::Mat& frame);
    
    ShaderProgram* getCurrentShader();
    
    // Callbacks
    static void errorCallback(int error, const char* description);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
};

