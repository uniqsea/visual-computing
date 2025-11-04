#include "utils/Timer.h"

Timer::Timer() 
    : deltaTime(0.0), frameTime(0.0), currentFPS(0.0f) {
    start();
}

void Timer::start() {
    startTime = Clock::now();
    lastFrameTime = startTime;
}

void Timer::update() {
    auto currentTime = Clock::now();
    
    // Calculate delta time
    std::chrono::duration<double> delta = currentTime - lastFrameTime;
    deltaTime = delta.count();
    frameTime = deltaTime * 1000.0; // Convert to milliseconds
    
    // Calculate FPS directly from current frame time
    if (deltaTime > 0.0) {
        currentFPS = static_cast<float>(1.0 / deltaTime);
    }
    
    lastFrameTime = currentTime;
}

