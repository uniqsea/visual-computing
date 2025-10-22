#pragma once

#include <GLFW/glfw3.h>
#include "filters/FilterManager.h"
#include <string>

class AffineTransform;
class PerformanceLogger;

struct GUIState {
    FilterType selectedFilter = FilterType::None;
    ProcessingMode processingMode = ProcessingMode::GPU;
    bool transformEnabled = false;
    int selectedResolution = 1; // 0: 640x480, 1: 1280x720, 2: 1920x1080
    
    // Filter parameters
    int pixelationBlockSize = 10;
    float cartoonEdgeThreshold = 50.0f;
    
    // Display
    float currentFPS = 0.0f;
    bool showDemoWindow = false;
};

class GUI {
public:
    GUI();
    ~GUI();
    
    bool initialize(GLFWwindow* window);
    void shutdown();
    
    void beginFrame();
    void render();
    void endFrame();
    
    void drawControlPanel(FilterManager& filterManager, 
                         AffineTransform& transform,
                         PerformanceLogger& perfLogger);
    
    void updateFPS(float fps);
    
    GUIState& getState() { return state; }
    const GUIState& getState() const { return state; }
    
    bool wantsMouseInput() const;
    bool wantsKeyboardInput() const;
    
private:
    GUIState state;
    std::string buildMode;
};

