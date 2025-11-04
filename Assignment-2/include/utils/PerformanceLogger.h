#pragma once

#include <string>
#include <vector>
#include <fstream>

struct PerformanceData {
    std::string resolution;
    std::string filter;
    std::string mode;        // CPU or GPU
    bool transformEnabled;
    std::string buildMode;   // Debug or Release
    
    // Aggregated averages (ms)
    float frameTimeAvgMs;   // End-to-end
    float algoTimeAvgMs;    // Algorithm-only
    
    int sampleCount;
    
    PerformanceData(const std::string& res, const std::string& filt,
                    const std::string& m, bool trans, const std::string& build,
                    float frameAvgMs, float algoAvgMs, int samples)
        : resolution(res), filter(filt), mode(m), transformEnabled(trans),
          buildMode(build), frameTimeAvgMs(frameAvgMs), algoTimeAvgMs(algoAvgMs),
          sampleCount(samples) {}
};

class PerformanceLogger {
public:
    PerformanceLogger();
    
    void addEntry(const PerformanceData& data);
    bool exportToCSV(const std::string& filename = "data/performance_results.csv");
    void clear();
    
    size_t getEntryCount() const { return entries.size(); }
    
private:
    std::vector<PerformanceData> entries;
};

