#pragma once

#include <memory>
#include "Renderer.h"
#include "VideoCapture.h"
#include "ShaderProgram.h"
#include "Texture.h"
#include "GUI.h"
#include "filters/FilterManager.h"
#include "transforms/AffineTransform.h"
#include "utils/Timer.h"
#include "utils/PerformanceLogger.h"
#include "utils/PerformanceEvaluation.h"

// Forward declaration
struct GLFWwindow;

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
    std::unique_ptr<PerformanceBenchmark> benchmark;
    
    // Shaders
    std::unique_ptr<ShaderProgram> basicShader;
    std::unique_ptr<ShaderProgram> pixelationShader;
    std::unique_ptr<ShaderProgram> cartoonShader;
    std::unique_ptr<ShaderProgram> oilPaintingShader;
    // Edge shader removed
    
    // State
    bool isRunning;
    cv::Mat currentFrame;
    std::string buildMode;
    double currentAlgoSec = 0.0; // measured per-frame algorithm processing time
    // Screenshot after benchmark completes
    bool pendingScreenshot = false;
    std::string pendingScreenshotPath;
    
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

    // Utilities
    void captureScreenshot(const std::string& absolutePath);
    
    // Callbacks
    static void errorCallback(int error, const char* description);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
};
