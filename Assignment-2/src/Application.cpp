#include "Application.h"
#include <iostream>
#include <glad/gl.h>

Application::Application() 
    : window(nullptr), windowWidth(1280), windowHeight(720),
      isRunning(false), leftMousePressed(false), rightMousePressed(false),
      lastMouseX(0.0), lastMouseY(0.0) {
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
    
    // Enable vsync
    glfwSwapInterval(1);
    
    // Initialize renderer
    renderer = std::make_unique<Renderer>();
    if (!renderer->initialize(width, height)) {
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
        applyFilter(currentFrame);
        applyTransform(currentFrame);
    }
    
    // Update texture
    videoTexture->updateFromMat(currentFrame);
}

void Application::renderFrame() {
    renderer->clear(0.1f, 0.1f, 0.1f, 1.0f);
    
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
            glm::mat4 transformMat = transform->getGPUMatrix();
            shader->setMat4("transform", transformMat);
        } else {
            shader->setMat4("transform", glm::mat4(1.0f));
        }
    }
    
    // Render textured quad
    renderer->renderTexturedQuad(*videoTexture, *shader);
}

void Application::updateGUI() {
    gui->beginFrame();
    gui->drawControlPanel(*filterManager, *transform, *perfLogger);
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
    
    while (!glfwWindowShouldClose(window) && isRunning) {
        // Update timer
        timer->update();
        
        // Poll events
        glfwPollEvents();
        
        // Handle input
        handleInput();
        
        // Process video frame
        processFrame();
        
        // Render
        renderFrame();
        
        // Update and render GUI
        gui->updateFPS(timer->getFPS());
        updateGUI();
        
        // Swap buffers
        glfwSwapBuffers(window);
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
            // Rotate (based on horizontal movement)
            float rotationSpeed = 0.01f;
            app->transform->rotate(static_cast<float>(dx) * rotationSpeed);
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
}

