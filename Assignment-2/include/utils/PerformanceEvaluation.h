#pragma once

#include <chrono>
#include <vector>
#include <string>

enum class BenchmarkState {
    Idle,       // Not running
    Warmup,     // 2-second warmup phase
    Recording,  // 10-second recording phase
    Complete    // Benchmark finished
};

struct BenchmarkResult {
    bool valid = false;
    
    // Configuration
    std::string resolution;
    std::string filter;
    std::string mode;
    bool transformEnabled;
    std::string buildMode;
    
    // Aggregated metrics (averages over recording samples)
    float frameTimeAvgMs = 0.0f;   // End-to-end frame time
    float algoTimeAvgMs = 0.0f;    // Algorithm processing time
    
    int sampleCount = 0;
    
    BenchmarkResult() = default;
};

class PerformanceBenchmark {
public:
    PerformanceBenchmark();
    
    // Start a new benchmark with current configuration
    void startBenchmark(const std::string& resolution, 
                       const std::string& filter,
                       const std::string& mode,
                       bool transformEnabled,
                       const std::string& buildMode);
    
    // Update benchmark state with frame times (called every frame)
    // totalFrameSec: end-to-end duration; algoFrameSec: algorithm processing time
    void update(double totalFrameSec, double algoFrameSec);
    
    // Query benchmark state
    bool isRunning() const;
    BenchmarkState getState() const { return state; }
    
    // Get elapsed time in current phase
    double getPhaseElapsedTime() const;
    double getPhaseTotalTime() const;
    
    // Get result after completion
    BenchmarkResult getResult() const { return result; }
    bool hasResult() const { return result.valid; }
    
    // Reset to idle state
    void reset();
    
private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    
    BenchmarkState state;
    TimePoint phaseStartTime;
    
    // Configuration for current benchmark
    std::string currentResolution;
    std::string currentFilter;
    std::string currentMode;
    bool currentTransformEnabled;
    std::string currentBuildMode;
    
    // Data collection (seconds)
    std::vector<double> totalFrameTimeSamples; // seconds
    std::vector<double> algoFrameTimeSamples;  // seconds
    
    // Result
    BenchmarkResult result;
    
    // Phase durations
    static constexpr double WARMUP_DURATION = 2.0;   // seconds
    static constexpr double RECORDING_DURATION = 10.0; // seconds
    
    // Helper methods
    void transitionToNextState();
    void calculateStatistics();
    static float computeAvgMs(const std::vector<double>& samples);
    double getElapsedTime() const;
};




