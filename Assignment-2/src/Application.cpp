#include "Application.h"
#include <GLFW/glfw3.h>
#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl3.h>
#include <filesystem>
#include <chrono>
#include <iostream>

Application::Application() 
    : window(nullptr), windowWidth(1280), windowHeight(720),
      isRunning(false), leftMousePressed(false), rightMousePressed(false),
      lastMouseX(0.0), lastMouseY(0.0) {
#ifdef DEBUG_BUILD
    buildMode = "Debug";
#else
    buildMode = "Release";
#endif
}

Application::~Application() {
    shutdown();
}

void Application::errorCallback(int error, const char* description) {
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

bool Application::initialize(int width, int height, const char* title) {
    windowWidth = width;
    windowHeight = height;
    
    // Initialize GLFW
    if (!initializeGLFW()) {
        return false;
    }
    
    // Create window
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    
    glfwMakeContextCurrent(window);
    glfwSetWindowUserPointer(window, this);
    
    // Setup callbacks
    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    
    // Disable vsync
    glfwSwapInterval(0);
    
    // Query actual framebuffer size (important on high-DPI displays)
    int framebufferWidth = 0;
    int framebufferHeight = 0;
    glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
    
    // Initialize renderer
    renderer = std::make_unique<Renderer>();
    if (!renderer->initialize(framebufferWidth, framebufferHeight)) {
        return false;
    }
    
    // Initialize GUI
    gui = std::make_unique<GUI>();
    if (!gui->initialize(window)) {
        return false;
    }
    
    // Initialize shaders
    if (!initializeShaders()) {
        return false;
    }
    
    // Initialize video capture
    if (!initializeVideoCapture()) {
        return false;
    }
    
    // Initialize other components
    videoTexture = std::make_unique<Texture>();
    filterManager = std::make_unique<FilterManager>();
    transform = std::make_unique<AffineTransform>();
    timer = std::make_unique<Timer>();
    perfLogger = std::make_unique<PerformanceLogger>();
    benchmark = std::make_unique<PerformanceBenchmark>();
    
    std::cout << "Application initialized successfully" << std::endl;
    isRunning = true;
    
    return true;
}

bool Application::initializeGLFW() {
    glfwSetErrorCallback(errorCallback);
    
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    
    // Set OpenGL version to 3.3 Core
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    
    return true;
}

bool Application::initializeShaders() {
    // Basic shader (passthrough)
    basicShader = std::make_unique<ShaderProgram>();
    if (!basicShader->loadFromFiles("shaders/basic.vert", "shaders/basic.frag")) {
        std::cerr << "Failed to load basic shader" << std::endl;
        return false;
    }
    
    // Pixelation shader
    pixelationShader = std::make_unique<ShaderProgram>();
    if (!pixelationShader->loadFromFiles("shaders/basic.vert", "shaders/pixelation.frag")) {
        std::cerr << "Failed to load pixelation shader" << std::endl;
        return false;
    }
    
    // Cartoon shader
    cartoonShader = std::make_unique<ShaderProgram>();
    if (!cartoonShader->loadFromFiles("shaders/basic.vert", "shaders/cartoon.frag")) {
        std::cerr << "Failed to load cartoon shader" << std::endl;
        return false;
    }
    
    // Oil painting shader
    oilPaintingShader = std::make_unique<ShaderProgram>();
    if (!oilPaintingShader->loadFromFiles("shaders/basic.vert", "shaders/oil_painting.frag")) {
        std::cerr << "Failed to load oil painting shader" << std::endl;
        return false;
    }

    // Edge shader removed
    
    return true;
}

bool Application::initializeVideoCapture() {
    videoCapture = std::make_unique<VideoCapture>();
    if (!videoCapture->open(0)) {
        std::cerr << "Failed to open video capture device" << std::endl;
        return false;
    }
    
    // Set initial resolution
    videoCapture->setResolution1280x720();
    
    return true;
}

ShaderProgram* Application::getCurrentShader() {
    auto filterType = filterManager->getCurrentFilterType();
    auto mode = filterManager->getProcessingMode();
    
    // If using CPU mode, always use basic shader
    if (mode == ProcessingMode::CPU) {
        return basicShader.get();
    }
    
    // GPU mode - select appropriate shader
    switch (filterType) {
        case FilterType::Pixelation:
            return pixelationShader.get();
        case FilterType::Cartoon:
            return cartoonShader.get();
        case FilterType::OilPainting:
            return oilPaintingShader.get();
        case FilterType::None:
        default:
            return basicShader.get();
    }
}

void Application::applyFilter(cv::Mat& frame) {
    auto mode = filterManager->getProcessingMode();
    auto* filter = filterManager->getCurrentFilter();
    
    if (mode == ProcessingMode::CPU && filter != nullptr) {
        filter->applyCPU(frame);
    }
    // GPU filters are applied in shaders during rendering
}

void Application::applyTransform(cv::Mat& frame) {
    if (gui->getState().transformEnabled) {
        auto mode = filterManager->getProcessingMode();
        if (mode == ProcessingMode::CPU) {
            transform->applyCPU(frame);
        }
        // GPU transform is applied in vertex shader during rendering
    }
}

void Application::processFrame() {
    // Capture frame
    if (!videoCapture->getFrame(currentFrame)) {
        std::cerr << "Failed to capture frame" << std::endl;
        return;
    }
    
    // Apply CPU processing if needed
    auto mode = filterManager->getProcessingMode();
    if (mode == ProcessingMode::CPU) {
        auto algoStart = std::chrono::high_resolution_clock::now();
        applyFilter(currentFrame);
        applyTransform(currentFrame);
        auto algoEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> algoDelta = algoEnd - algoStart;
        currentAlgoSec = algoDelta.count();
    }

    // Update base texture (mirror option toggled in GUI)
    // Invert checkbox semantics: unchecked behaves like previous "checked"
    videoTexture->updateFromMat(currentFrame, !gui->getState().mirrorPreview);

    // six-view grid removed
}

void Application::renderFrame() {
    renderer->clear(0.1f, 0.1f, 0.1f, 1.0f);
    // If no valid texture has been uploaded yet (e.g., camera not ready), skip rendering
    if (!videoTexture || videoTexture->getID() == 0) {
        return;
    }
    
    {
        // Original single-view rendering
        // Get appropriate shader
        ShaderProgram* shader = getCurrentShader();
        shader->use();
        
        // Set uniforms
        shader->setInt("videoTexture", 0);
        
        // Apply GPU filter parameters if needed
        auto mode = filterManager->getProcessingMode();
        if (mode == ProcessingMode::GPU) {
            auto* filter = filterManager->getCurrentFilter();
            if (filter != nullptr) {
                filter->applyGPU(*shader);
            }
            
            // Apply transform matrix if enabled
            if (gui->getState().transformEnabled) {
                // Match transform to the actual draw viewport size (letterbox)
                int fbw_local = renderer->getWidth();
                int fbh_local = renderer->getHeight();
                int regionW_local = fbw_local - static_cast<int>(gui->getPanelWidth());
                if (regionW_local < 1) regionW_local = fbw_local;
                int texW_local = videoTexture->getWidth();
                int texH_local = videoTexture->getHeight();
                int drawW_local = regionW_local;
                int drawH_local = fbh_local;
                if (texW_local > 0 && texH_local > 0) {
                    double rRegion_local = static_cast<double>(regionW_local) / static_cast<double>(fbh_local);
                    double rVideo_local  = static_cast<double>(texW_local) / static_cast<double>(texH_local);
                    if (rVideo_local > rRegion_local) {
                        drawW_local = regionW_local;
                        drawH_local = static_cast<int>(static_cast<double>(drawW_local) / rVideo_local);
                    } else {
                        drawH_local = fbh_local;
                        drawW_local = static_cast<int>(static_cast<double>(drawH_local) * rVideo_local);
                    }
                }
                glm::mat4 transformMat = transform->getGPUMatrixForViewport(static_cast<float>(drawW_local), static_cast<float>(drawH_local));
                shader->setMat4("transform", transformMat);
            } else {
                shader->setMat4("transform", glm::mat4(1.0f));
            }
        }
        
        // Measure algorithm time in GPU mode around the draw call
        auto algoStart = std::chrono::high_resolution_clock::now();
        
        // Render textured quad into left region with aspect-preserving letterbox
        int fbw = renderer->getWidth();
        int fbh = renderer->getHeight();
        int panelPx = static_cast<int>(gui->getPanelWidth());
        int regionW = fbw - panelPx;
        if (regionW < 1) regionW = fbw; // fallback safety

        int texW = videoTexture->getWidth();
        int texH = videoTexture->getHeight();

        // Letterbox to match selected resolution/video aspect, do not crop/fill
        int drawW = regionW;
        int drawH = fbh;
        int offsetX = 0;
        int offsetY = 0;
        if (texW > 0 && texH > 0) {
            double rRegion = static_cast<double>(regionW) / static_cast<double>(fbh);
            double rVideo  = static_cast<double>(texW) / static_cast<double>(texH);
            if (rVideo > rRegion) {
                // Fit width
                drawW = regionW;
                drawH = static_cast<int>(static_cast<double>(drawW) / rVideo);
                offsetY = (fbh - drawH) / 2;
            } else {
                // Fit height
                drawH = fbh;
                drawW = static_cast<int>(static_cast<double>(drawH) * rVideo);
                offsetX = (regionW - drawW) / 2;
            }
        }
        glViewport(offsetX, offsetY, drawW, drawH);
        shader->setVec2("uvScale", glm::vec2(1.0f, 1.0f));
        // For GPU Pixelation, switch to nearest magnification to match CPU's sharp block edges
        bool restoreMagLinear = false;
        if (mode == ProcessingMode::GPU && filterManager->getCurrentFilterType() == FilterType::Pixelation) {
            glBindTexture(GL_TEXTURE_2D, videoTexture->getID());
            GLint prevMag = 0;
            glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, &prevMag);
            if (prevMag != GL_NEAREST) {
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                restoreMagLinear = true;
            }
        }

        renderer->renderTexturedQuad(*videoTexture, *shader);

        if (restoreMagLinear) {
            glBindTexture(GL_TEXTURE_2D, videoTexture->getID());
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }
        if (mode == ProcessingMode::GPU) {
            glFinish();
            auto algoEnd = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> algoDelta = algoEnd - algoStart;
            currentAlgoSec = algoDelta.count();
        }
        // Restore full viewport for subsequent GUI rendering
        glViewport(0, 0, fbw, fbh);
    }
}

void Application::captureScreenshot(const std::string& absolutePath) {
    int fbw = renderer->getWidth();
    int fbh = renderer->getHeight();
    if (fbw <= 0 || fbh <= 0) return;
    // Read from back buffer (current frame prior to swap)
    std::vector<unsigned char> rgba(static_cast<size_t>(fbw) * static_cast<size_t>(fbh) * 4);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, fbw, fbh, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    
    // Convert to OpenCV Mat, flip vertically, convert RGBA->BGR
    cv::Mat img(fbh, fbw, CV_8UC4, rgba.data());
    cv::Mat flipped; cv::flip(img, flipped, 0);
    cv::Mat bgr; cv::cvtColor(flipped, bgr, cv::COLOR_RGBA2BGR);
    
    try {
        namespace fs = std::filesystem;
        fs::path p(absolutePath);
        if (p.has_parent_path()) fs::create_directories(p.parent_path());
        if (cv::imwrite(absolutePath, bgr)) {
            std::cout << "Screenshot saved to: " << absolutePath << std::endl;
        } else {
            std::cerr << "Failed to save screenshot to: " << absolutePath << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Screenshot error: " << e.what() << std::endl;
    }
}

void Application::updateGUI() {
    gui->beginFrame();
    gui->drawControlPanel(*filterManager, *transform, *perfLogger, *benchmark);
    gui->endFrame();
    gui->render();
}

void Application::handleInput() {
    // Check for resolution changes
    auto& state = gui->getState();
    static int lastResolution = state.selectedResolution;
    
    if (state.selectedResolution != lastResolution) {
        switch (state.selectedResolution) {
            case 0:
                videoCapture->setResolution640x480();
                break;
            case 1:
                videoCapture->setResolution1280x720();
                break;
            case 2:
                videoCapture->setResolution1920x1080();
                break;
        }
        lastResolution = state.selectedResolution;
    }
}

void Application::run() {
    timer->start();
    
    // Helper to get resolution string
    auto getResolutionString = [this]() -> std::string {
        switch (gui->getState().selectedResolution) {
            case 0: return "640x480";
            case 1: return "1280x720";
            case 2: return "1920x1080";
            default: return "Unknown";
        }
    };
    
    // Helper to get filter string
    auto getFilterString = [this]() -> std::string {
        switch (gui->getState().selectedFilter) {
            case FilterType::None: return "None";
            case FilterType::Pixelation: return "Pixelation";
        case FilterType::Cartoon: return "Comic Art";
            case FilterType::OilPainting: return "OilPainting";
            default: return "Unknown";
        }
    };
    
    // Helper to get mode string
    auto getModeString = [this]() -> std::string {
        return (gui->getState().processingMode == ProcessingMode::CPU) ? "CPU" : "GPU";
    };
    
    while (!glfwWindowShouldClose(window) && isRunning) {
        // Update timer
        timer->update();
        
        // Poll events
        glfwPollEvents();
        
        // Handle input (skip configuration changes during Recording)
        if (benchmark->getState() != BenchmarkState::Recording) {
            handleInput();
        }
        
        // Check if benchmark start is requested
        auto& guiState = gui->getState();
        if (guiState.startBenchmarkRequested) {
            guiState.startBenchmarkRequested = false;
            benchmark->startBenchmark(
                getResolutionString(),
                getFilterString(),
                getModeString(),
                guiState.transformEnabled,
                buildMode
            );
        }
        
        // Update benchmark with current TOTAL frame time; pipeline added after swap
        double totalFrameSec = timer->getDeltaTime();
        
        // Check if benchmark just completed
        static BenchmarkState lastBenchmarkState = BenchmarkState::Idle;
        BenchmarkState currentBenchmarkState = benchmark->getState();
        if (lastBenchmarkState == BenchmarkState::Recording && 
            currentBenchmarkState == BenchmarkState::Complete) {
            // Benchmark just finished - save results
            if (benchmark->hasResult()) {
                BenchmarkResult result = benchmark->getResult();
                PerformanceData data(
                    result.resolution,
                    result.filter,
                    result.mode,
                    result.transformEnabled,
                    result.buildMode,
                    result.frameTimeAvgMs,
                    result.algoTimeAvgMs,
                    result.sampleCount
                );
                perfLogger->addEntry(data);
                std::cout << "Performance evaluation results automatically saved to logger." << std::endl;
                // Auto-export results to CSV after each evaluation completes (absolute path)
                const char* kExportAbsPath = "/Users/uniqsea/Workspace/visual-computing/Assignment-2/data/performance_results.csv";
                if (perfLogger->exportToCSV(kExportAbsPath)) {
                    std::cout << "Exported performance data to " << kExportAbsPath << std::endl;
                } else {
                    std::cerr << "Failed to export performance data (see previous errors for paths)" << std::endl;
                }

                // Queue a screenshot of the full window (with GUI visible) on this frame
                auto resSuffix = [this]() -> std::string {
                    switch (gui->getState().selectedResolution) {
                        case 0: return "480p";
                        case 1: return "720p";
                        case 2: return "1080p";
                        default: return "Unknown";
                    }
                }();
                std::string filename = resSuffix + "-" + getFilterString() + "-" + getModeString() + "-" + (gui->getState().transformEnabled ? std::string("ON") : std::string("OFF")) + "-" + buildMode + ".png";
                pendingScreenshotPath = std::string("/Users/uniqsea/Workspace/visual-computing/Assignment-2/data/") + filename;
                pendingScreenshot = true;
            }
        }
        lastBenchmarkState = currentBenchmarkState;
        
        // Pipeline timing window: process → render → swap (exclude GUI)
        auto pipelineStart = std::chrono::high_resolution_clock::now();

        // Process video frame
        processFrame();
        
        // Render
        renderFrame();
        
        // Always hide GUI during Recording for pure timing
        if (benchmark->getState() != BenchmarkState::Recording) {
            gui->updateFPS(timer->getFPS());
            gui->updateFrameTime(static_cast<float>(timer->getFrameTime()));
            updateGUI();
        }

        // If a screenshot was requested (after benchmark complete), capture now before swap
        if (pendingScreenshot) {
            captureScreenshot(pendingScreenshotPath);
            pendingScreenshot = false;
        }
        
        // Swap buffers (Display)
        glfwSwapBuffers(window);

        auto pipelineEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> pipelineDelta = pipelineEnd - pipelineStart;
        benchmark->update(totalFrameSec, currentAlgoSec);
    }
}

void Application::shutdown() {
    if (gui) {
        gui->shutdown();
    }
    
    if (renderer) {
        renderer->shutdown();
    }
    
    if (videoCapture) {
        videoCapture->close();
    }
    
    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    
    glfwTerminate();
    
    std::cout << "Application shutdown complete" << std::endl;
}

// Callbacks implementation
void Application::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
    }
}

void Application::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    
    // Don't process mouse if GUI wants it
    if (app->gui->wantsMouseInput()) {
        return;
    }
    
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        app->leftMousePressed = (action == GLFW_PRESS);
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        app->rightMousePressed = (action == GLFW_PRESS);
    }
}

void Application::cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    
    // Don't process mouse if GUI wants it
    if (app->gui->wantsMouseInput()) {
        return;
    }
    
    double dx = xpos - app->lastMouseX;
    double dy = ypos - app->lastMouseY;
    
    if (app->gui->getState().transformEnabled) {
        if (app->leftMousePressed) {
            // Translate
            app->transform->translate(static_cast<float>(dx), static_cast<float>(dy));
        } else if (app->rightMousePressed) {
            // Rotate (based on horizontal movement); right-drag → clockwise
            float rotationSpeed = 0.01f;
            app->transform->rotate(static_cast<float>(-dx) * rotationSpeed);
        }
    }
    
    app->lastMouseX = xpos;
    app->lastMouseY = ypos;
}

void Application::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    
    // Don't process mouse if GUI wants it
    if (app->gui->wantsMouseInput()) {
        return;
    }
    
    if (app->gui->getState().transformEnabled) {
        float scaleFactor = 1.0f + static_cast<float>(yoffset) * 0.1f;
        app->transform->scale(scaleFactor);
    }
}

void Application::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    // Keep renderer's notion of size in sync so multi-view layout scales with window
    if (auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window))) {
        if (app->renderer) {
            app->renderer->updateSize(width, height);
        }
    }
}
