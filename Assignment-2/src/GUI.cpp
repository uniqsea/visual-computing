#include "GUI.h"
#include "transforms/AffineTransform.h"
#include "utils/PerformanceLogger.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>

GUI::GUI() {
#ifdef DEBUG_BUILD
    buildMode = "Debug";
#else
    buildMode = "Release";
#endif
}

GUI::~GUI() {
    shutdown();
}

bool GUI::initialize(GLFWwindow* window) {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    
    // Setup Platform/Renderer backends
    const char* glsl_version = "#version 330";
    if (!ImGui_ImplGlfw_InitForOpenGL(window, true)) {
        std::cerr << "Failed to initialize ImGui GLFW backend" << std::endl;
        return false;
    }
    
    if (!ImGui_ImplOpenGL3_Init(glsl_version)) {
        std::cerr << "Failed to initialize ImGui OpenGL3 backend" << std::endl;
        return false;
    }
    
    std::cout << "GUI initialized successfully" << std::endl;
    return true;
}

void GUI::shutdown() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void GUI::beginFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void GUI::endFrame() {
    ImGui::Render();
}

void GUI::render() {
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void GUI::drawControlPanel(FilterManager& filterManager, 
                          AffineTransform& transform,
                          PerformanceLogger& perfLogger) {
    ImGui::Begin("Real-time Video Processing", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    
    // FPS Display
    ImGui::Text("FPS: %.1f (%.2f ms/frame)", state.currentFPS, 1000.0f / state.currentFPS);
    ImGui::Text("Build Mode: %s", buildMode.c_str());
    ImGui::Separator();
    
    // Filter Selection
    ImGui::Text("Filter Selection");
    const char* filterNames[] = { "None", "Pixelation", "Cartoon" };
    int currentFilter = static_cast<int>(state.selectedFilter);
    if (ImGui::Combo("Filter", &currentFilter, filterNames, 3)) {
        state.selectedFilter = static_cast<FilterType>(currentFilter);
        filterManager.setCurrentFilter(state.selectedFilter);
    }
    
    // Processing Mode
    ImGui::Text("Processing Mode");
    const char* modeNames[] = { "CPU", "GPU" };
    int currentMode = (state.processingMode == ProcessingMode::CPU) ? 0 : 1;
    if (ImGui::Combo("Mode", &currentMode, modeNames, 2)) {
        state.processingMode = (currentMode == 0) ? ProcessingMode::CPU : ProcessingMode::GPU;
        filterManager.setProcessingMode(state.processingMode);
    }
    
    ImGui::Separator();
    
    // Filter Parameters
    if (state.selectedFilter == FilterType::Pixelation) {
        ImGui::Text("Pixelation Parameters");
        if (ImGui::SliderInt("Block Size", &state.pixelationBlockSize, 2, 50)) {
            if (auto* filter = dynamic_cast<PixelationFilter*>(filterManager.getCurrentFilter())) {
                filter->setBlockSize(state.pixelationBlockSize);
            }
        }
    } else if (state.selectedFilter == FilterType::Cartoon) {
        ImGui::Text("Cartoon Parameters");
        ImGui::SliderFloat("Edge Threshold", &state.cartoonEdgeThreshold, 10.0f, 200.0f);
    }
    
    ImGui::Separator();
    
    // Resolution Selection
    ImGui::Text("Resolution");
    const char* resolutions[] = { "640x480", "1280x720", "1920x1080" };
    ImGui::Combo("Resolution", &state.selectedResolution, resolutions, 3);
    
    ImGui::Separator();
    
    // Geometric Transform
    ImGui::Text("Geometric Transform");
    ImGui::Checkbox("Enable Transform", &state.transformEnabled);
    
    if (state.transformEnabled) {
        ImGui::Text("Controls:");
        ImGui::BulletText("Left Mouse: Drag to translate");
        ImGui::BulletText("Right Mouse: Drag to rotate");
        ImGui::BulletText("Mouse Wheel: Scale");
        
        if (ImGui::Button("Reset Transform")) {
            transform.reset();
        }
        
        ImGui::Text("Current Values:");
        glm::vec2 trans = transform.getTranslation();
        ImGui::Text("  Translation: (%.1f, %.1f)", trans.x, trans.y);
        ImGui::Text("  Rotation: %.2f deg", transform.getRotation() * 180.0f / 3.14159f);
        ImGui::Text("  Scale: %.2f", transform.getScale());
    }
    
    ImGui::Separator();
    
    // Performance Logging
    ImGui::Text("Performance Logging");
    if (ImGui::Button("Export Performance Data")) {
        perfLogger.exportToCSV();
    }
    ImGui::SameLine();
    ImGui::Text("Entries: %zu", perfLogger.getEntryCount());
    
    if (ImGui::Button("Clear Performance Data")) {
        perfLogger.clear();
    }
    
    ImGui::Separator();
    
    // Demo Window Toggle
    ImGui::Checkbox("Show ImGui Demo", &state.showDemoWindow);
    
    ImGui::End();
    
    // Show demo window if enabled
    if (state.showDemoWindow) {
        ImGui::ShowDemoWindow(&state.showDemoWindow);
    }
}

void GUI::updateFPS(float fps) {
    state.currentFPS = fps;
}

bool GUI::wantsMouseInput() const {
    ImGuiIO& io = ImGui::GetIO();
    return io.WantCaptureMouse;
}

bool GUI::wantsKeyboardInput() const {
    ImGuiIO& io = ImGui::GetIO();
    return io.WantCaptureKeyboard;
}

