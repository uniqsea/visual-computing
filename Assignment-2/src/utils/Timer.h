#pragma once

#include <chrono>
#include <vector>

class Timer {
public:
    Timer();
    
    void start();
    void update();
    
    float getFPS() const { return currentFPS; }
    double getFrameTime() const { return frameTime; }
    double getDeltaTime() const { return deltaTime; }
    
private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    
    TimePoint startTime;
    TimePoint lastFrameTime;
    
    double deltaTime;
    double frameTime;
    float currentFPS;
    
    // Rolling average for FPS calculation
    std::vector<double> frameTimes;
    size_t maxSamples = 60;
    size_t currentIndex = 0;
};

