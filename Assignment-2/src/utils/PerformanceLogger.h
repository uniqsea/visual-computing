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
    float fps;
    float stdDev;
    
    PerformanceData(const std::string& res, const std::string& filt, 
                   const std::string& m, bool trans, const std::string& build,
                   float f, float std = 0.0f)
        : resolution(res), filter(filt), mode(m), transformEnabled(trans),
          buildMode(build), fps(f), stdDev(std) {}
};

class PerformanceLogger {
public:
    PerformanceLogger();
    
    void addEntry(const PerformanceData& data);
    void exportToCSV(const std::string& filename = "data/performance_results.csv");
    void clear();
    
    size_t getEntryCount() const { return entries.size(); }
    
private:
    std::vector<PerformanceData> entries;
};

