#include "GUI.h"
#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl3.h>
#include <GLFW/glfw3.h>
#include "transforms/AffineTransform.h"
#include "utils/PerformanceLogger.h"
#include "utils/PerformanceEvaluation.h"
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
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 10.0f;
    style.FrameRounding = 7.0f;
    style.GrabRounding = 7.0f;
    style.WindowPadding = ImVec2(18.0f, 18.0f);
    style.FramePadding = ImVec2(12.0f, 8.0f);
    style.ItemSpacing = ImVec2(12.0f, 10.0f);
    style.ScrollbarSize = 14.0f;
    
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg]        = ImVec4(0.10f, 0.11f, 0.14f, 0.95f);
    colors[ImGuiCol_Header]          = ImVec4(0.20f, 0.39f, 0.58f, 1.00f);
    colors[ImGuiCol_HeaderHovered]   = ImVec4(0.26f, 0.51f, 0.73f, 1.00f);
    colors[ImGuiCol_HeaderActive]    = ImVec4(0.30f, 0.55f, 0.80f, 1.00f);
    colors[ImGuiCol_Button]          = ImVec4(0.20f, 0.37f, 0.57f, 1.00f);
    colors[ImGuiCol_ButtonHovered]   = ImVec4(0.26f, 0.51f, 0.73f, 1.00f);
    colors[ImGuiCol_ButtonActive]    = ImVec4(0.30f, 0.58f, 0.82f, 1.00f);
    colors[ImGuiCol_FrameBg]         = ImVec4(0.17f, 0.21f, 0.28f, 1.00f);
    colors[ImGuiCol_FrameBgHovered]  = ImVec4(0.26f, 0.36f, 0.49f, 1.00f);
    colors[ImGuiCol_FrameBgActive]   = ImVec4(0.29f, 0.46f, 0.67f, 1.00f);
    colors[ImGuiCol_SliderGrab]      = ImVec4(0.34f, 0.60f, 0.86f, 1.00f);
    colors[ImGuiCol_SliderGrabActive]= ImVec4(0.42f, 0.71f, 0.97f, 1.00f);
    colors[ImGuiCol_CheckMark]       = ImVec4(0.70f, 0.88f, 1.00f, 1.00f);
    colors[ImGuiCol_TitleBg]         = ImVec4(0.11f, 0.15f, 0.21f, 1.00f);
    colors[ImGuiCol_TitleBgActive]   = ImVec4(0.16f, 0.27f, 0.40f, 1.00f);
    colors[ImGuiCol_PopupBg]         = ImVec4(0.09f, 0.10f, 0.13f, 0.94f);
    
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
                          PerformanceLogger& perfLogger,
                          PerformanceBenchmark& benchmark) {
    // Anchor control panel to the right side of the window
    ImGuiIO& io_for_layout = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(io_for_layout.DisplaySize.x - panelWidth, 0.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(panelWidth, io_for_layout.DisplaySize.y), ImGuiCond_Always);

    ImGuiWindowFlags dockFlags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize;
    ImGui::Begin("Real-time Video Processing", nullptr, dockFlags);
    
    const ImVec4 accent = ImVec4(0.66f, 0.85f, 1.00f, 1.00f);
    
    // FPS Display
    ImGui::TextColored(accent, "Performance");
    ImGui::SameLine();
    if (benchmark.getState() == BenchmarkState::Idle) {
        if (ImGui::Button("Start Evaluation")) {
            state.startBenchmarkRequested = true;
        }
    }
    ImGui::Spacing();
    ImGui::Text("FPS: %.1f", state.currentFPS);
    ImGui::Text("Frame Time: %.2f ms", state.currentFrameTimeMs);
    ImGui::Text("Build Mode: %s", buildMode.c_str());
    ImGui::Separator();
    
    // Filter Selection
    ImGui::TextColored(accent, "Filters");
    const char* filterNames[] = { "None", "Pixelation", "Comic Art", "OilPainting" };
    int currentFilter = static_cast<int>(state.selectedFilter);
    if (ImGui::Combo("##filter_combo", &currentFilter, filterNames, 4)) {
        state.selectedFilter = static_cast<FilterType>(currentFilter);
        filterManager.setCurrentFilter(state.selectedFilter);
    }
    
    // Processing Mode
    ImGui::Spacing();
    ImGui::TextColored(accent, "Backend");
    const char* modeNames[] = { "CPU", "GPU" };
    int currentMode = (state.processingMode == ProcessingMode::CPU) ? 0 : 1;
    if (ImGui::Combo("##processing_mode_combo", &currentMode, modeNames, 2)) {
        state.processingMode = (currentMode == 0) ? ProcessingMode::CPU : ProcessingMode::GPU;
        filterManager.setProcessingMode(state.processingMode);
    }
    
    ImGui::Separator();
    
    // Filter Parameters
    if (state.selectedFilter == FilterType::Pixelation) {
        ImGui::Text("Pixelation Parameters");
        if (ImGui::SliderInt("Block Size", &state.pixelationBlockSize, 1, 50)) {
            if (auto* filter = dynamic_cast<PixelationFilter*>(filterManager.getCurrentFilter())) {
                filter->setBlockSize(state.pixelationBlockSize);
            }
        }
    } else if (state.selectedFilter == FilterType::Cartoon) {
        ImGui::Text("Comic Art Parameters");
        if (ImGui::SliderFloat("Edge Threshold", &state.cartoonEdgeThreshold, 10.0f, 100.0f)) {
            if (auto* filter = dynamic_cast<CartoonFilter*>(filterManager.getCurrentFilter())) {
                filter->setParameter("edgeThreshold", state.cartoonEdgeThreshold);
            }
        }
    } else if (state.selectedFilter == FilterType::OilPainting) {
        ImGui::Text("Oil Painting Parameters");
        if (ImGui::SliderInt("Brush Radius", &state.oilPaintingRadius, 1, 10)) {
            if (auto* filter = dynamic_cast<OilPaintingFilter*>(filterManager.getCurrentFilter())) {
                filter->setParameter("radius", static_cast<float>(state.oilPaintingRadius));
            }
        }
        // Intensity removed: single-parameter (Brush Radius) design
    }
    
    ImGui::Separator();
    
    // Resolution Selection
    ImGui::Spacing();
    ImGui::TextColored(accent, "Resolution");
    const char* resolutions[] = { "640x480", "1280x720", "1920x1080" };
    ImGui::Combo("##resolution_combo", &state.selectedResolution, resolutions, 3);
    
    ImGui::Separator();
    
    // Display Options
    ImGui::Spacing();
    ImGui::TextColored(accent, "Display");
    ImGui::Checkbox("Mirror Preview", &state.mirrorPreview);
    // Six-view grid removed to restore single-view rendering
    
    ImGui::Separator();
    
    // Geometric Transform
    ImGui::Spacing();
    ImGui::TextColored(accent, "Geometric Transform");
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
    
    // Inline performance evaluation status (no separate section)
    BenchmarkState benchmarkState = benchmark.getState();
    if (benchmarkState == BenchmarkState::Warmup) {
        ImGui::Text("Warming up...");
        double elapsed = benchmark.getPhaseElapsedTime();
        double total = benchmark.getPhaseTotalTime();
        float progress = static_cast<float>(elapsed / total);
        ImGui::ProgressBar(progress, ImVec2(-1, 0));
        ImGui::Text("%.1f / %.1f seconds", elapsed, total);
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.2f, 1.0f), "GUI will be hidden in Recording");
    } else if (benchmarkState == BenchmarkState::Recording) {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "RECORDING");
        double elapsed = benchmark.getPhaseElapsedTime();
        double total = benchmark.getPhaseTotalTime();
        float progress = static_cast<float>(elapsed / total);
        ImGui::ProgressBar(progress, ImVec2(-1, 0));
        ImGui::Text("%.1f / %.1f seconds", elapsed, total);
        ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.2f, 1.0f), "GUI hidden for pure pipeline timing");
    } else if (benchmarkState == BenchmarkState::Complete) {
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "Evaluation Complete");
        if (benchmark.hasResult()) {
            BenchmarkResult result = benchmark.getResult();
            ImGui::Text("Averages (ms)");
            ImGui::BulletText("Frame Time (end-to-end): %.2f ms", result.frameTimeAvgMs);
            ImGui::BulletText("Algorithm Time: %.2f ms", result.algoTimeAvgMs);
            ImGui::BulletText("Samples: %d", result.sampleCount);
        }
        if (ImGui::Button("Reset")) {
            benchmark.reset();
        }
    }
    
    ImGui::End();
}

void GUI::updateFPS(float fps) {
    state.currentFPS = fps;
}

void GUI::updateFrameTime(float frameTimeMs) {
    state.currentFrameTimeMs = frameTimeMs;
}

bool GUI::wantsMouseInput() const {
    ImGuiIO& io = ImGui::GetIO();
    return io.WantCaptureMouse;
}

bool GUI::wantsKeyboardInput() const {
    ImGuiIO& io = ImGui::GetIO();
    return io.WantCaptureKeyboard;
}
