#include "Timer.h"
#include <numeric>
#include <algorithm>

Timer::Timer() 
    : deltaTime(0.0), frameTime(0.0), currentFPS(0.0f) {
    frameTimes.resize(maxSamples, 0.0);
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
    
    // Update rolling average
    frameTimes[currentIndex] = deltaTime;
    currentIndex = (currentIndex + 1) % maxSamples;
    
    // Calculate average FPS
    double avgFrameTime = std::accumulate(frameTimes.begin(), frameTimes.end(), 0.0) / maxSamples;
    if (avgFrameTime > 0.0) {
        currentFPS = static_cast<float>(1.0 / avgFrameTime);
    }
    
    lastFrameTime = currentTime;
}

