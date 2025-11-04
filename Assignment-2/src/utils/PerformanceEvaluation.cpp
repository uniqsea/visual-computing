#include "utils/PerformanceEvaluation.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

PerformanceBenchmark::PerformanceBenchmark() 
    : state(BenchmarkState::Idle) {
}

void PerformanceBenchmark::startBenchmark(const std::string& resolution,
                                         const std::string& filter,
                                         const std::string& mode,
                                         bool transformEnabled,
                                         const std::string& buildMode) {
    // Store configuration
    currentResolution = resolution;
    currentFilter = filter;
    currentMode = mode;
    currentTransformEnabled = transformEnabled;
    currentBuildMode = buildMode;
    
    // Clear previous data
    totalFrameTimeSamples.clear();
    algoFrameTimeSamples.clear();
    result = BenchmarkResult();
    
    // Start warmup phase
    state = BenchmarkState::Warmup;
    phaseStartTime = Clock::now();
    
    std::cout << "Performance evaluation started - Warmup phase (2s)..." << std::endl;
    std::cout << "Configuration: " << resolution << " | " << filter 
              << " | " << mode << " | Transform: " << (transformEnabled ? "ON" : "OFF")
              << " | Build: " << buildMode << std::endl;
}

void PerformanceBenchmark::update(double totalFrameSec, double algoFrameSec) {
    if (state == BenchmarkState::Idle || state == BenchmarkState::Complete) {
        return;
    }
    
    double elapsed = getElapsedTime();
    
    switch (state) {
        case BenchmarkState::Warmup:
            if (elapsed >= WARMUP_DURATION) {
                transitionToNextState();
            }
            break;
            
        case BenchmarkState::Recording:
            // Collect both total and algorithm times
            totalFrameTimeSamples.push_back(totalFrameSec);
            algoFrameTimeSamples.push_back(algoFrameSec);
            
            if (elapsed >= RECORDING_DURATION) {
                transitionToNextState();
            }
            break;
            
        default:
            break;
    }
}

bool PerformanceBenchmark::isRunning() const {
    return state == BenchmarkState::Warmup || state == BenchmarkState::Recording;
}

double PerformanceBenchmark::getPhaseElapsedTime() const {
    if (state == BenchmarkState::Idle || state == BenchmarkState::Complete) {
        return 0.0;
    }
    return getElapsedTime();
}

double PerformanceBenchmark::getPhaseTotalTime() const {
    switch (state) {
        case BenchmarkState::Warmup:
            return WARMUP_DURATION;
        case BenchmarkState::Recording:
            return RECORDING_DURATION;
        default:
            return 0.0;
    }
}

void PerformanceBenchmark::reset() {
    state = BenchmarkState::Idle;
    totalFrameTimeSamples.clear();
    algoFrameTimeSamples.clear();
    result = BenchmarkResult();
}

void PerformanceBenchmark::transitionToNextState() {
    switch (state) {
        case BenchmarkState::Warmup:
            state = BenchmarkState::Recording;
            phaseStartTime = Clock::now();
            totalFrameTimeSamples.clear();
            algoFrameTimeSamples.clear();
            std::cout << "Performance evaluation recording phase started (10s)..." << std::endl;
            break;
            
        case BenchmarkState::Recording:
            state = BenchmarkState::Complete;
            calculateStatistics();
            std::cout << "Performance evaluation complete!" << std::endl;
            break;
            
        default:
            break;
    }
}

void PerformanceBenchmark::calculateStatistics() {
    if (totalFrameTimeSamples.empty() || algoFrameTimeSamples.empty()) {
        std::cerr << "Warning: No frame samples collected!" << std::endl;
        return;
    }
    
    // Store configuration in result
    result.resolution = currentResolution;
    result.filter = currentFilter;
    result.mode = currentMode;
    result.transformEnabled = currentTransformEnabled;
    result.buildMode = currentBuildMode;
    result.sampleCount = static_cast<int>(algoFrameTimeSamples.size());
    
    // Compute averages (ms)
    result.frameTimeAvgMs = computeAvgMs(totalFrameTimeSamples);
    result.algoTimeAvgMs = computeAvgMs(algoFrameTimeSamples);
    
    result.valid = true;
}

float PerformanceBenchmark::computeAvgMs(const std::vector<double>& samples) {
    if (samples.empty()) return 0.0f;
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    double avgSec = sum / static_cast<double>(samples.size());
    return static_cast<float>(avgSec * 1000.0);
}

double PerformanceBenchmark::getElapsedTime() const {
    auto now = Clock::now();
    std::chrono::duration<double> elapsed = now - phaseStartTime;
    return elapsed.count();
}


