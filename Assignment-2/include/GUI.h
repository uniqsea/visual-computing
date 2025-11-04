#pragma once

#include "filters/FilterManager.h"
#include <string>

// Forward declarations
struct GLFWwindow;
class AffineTransform;
class PerformanceLogger;
class PerformanceBenchmark;

struct GUIState {
    FilterType selectedFilter = FilterType::None;
    ProcessingMode processingMode = ProcessingMode::GPU;
    bool transformEnabled = false;
    int selectedResolution = 1; // 0: 640x480, 1: 1280x720, 2: 1920x1080
    bool mirrorPreview = false; // unchecked means behave like previous "checked"
    
    // Filter parameters
    int pixelationBlockSize = 25;      // 1..50, midpoint
    float cartoonEdgeThreshold = 55.0f;  // 10..100, midpoint
    int   oilPaintingRadius = 6;       // 1..10, midpoint
    int   oilPaintingIntensity = 24;   // unused slider, keep midpoint
    // Edge filter removed
    
    // Display
    float currentFPS = 0.0f;
    float currentFrameTimeMs = 0.0f;
    
    // Benchmark
    bool startBenchmarkRequested = false;
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
                         PerformanceLogger& perfLogger,
                         PerformanceBenchmark& benchmark);
    
    void updateFPS(float fps);
    void updateFrameTime(float frameTimeMs);
    
    GUIState& getState() { return state; }
    const GUIState& getState() const { return state; }
    
    bool wantsMouseInput() const;
    bool wantsKeyboardInput() const;
    
private:
    GUIState state;
    std::string buildMode;
    float panelWidth = 360.0f; // fixed right-side panel width
public:
    float getPanelWidth() const { return panelWidth; }
};
