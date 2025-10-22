#include "PerformanceLogger.h"
#include <iostream>
#include <iomanip>

PerformanceLogger::PerformanceLogger() {
}

void PerformanceLogger::addEntry(const PerformanceData& data) {
    entries.push_back(data);
    std::cout << "Logged: " << data.resolution << " | " << data.filter 
              << " | " << data.mode << " | Transform: " << (data.transformEnabled ? "ON" : "OFF")
              << " | " << data.buildMode << " | FPS: " << std::fixed 
              << std::setprecision(2) << data.fps << std::endl;
}

void PerformanceLogger::exportToCSV(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "Resolution,Filter,Mode,Transform,BuildMode,FPS,StdDev\n";
    
    // Write data
    for (const auto& entry : entries) {
        file << entry.resolution << ","
             << entry.filter << ","
             << entry.mode << ","
             << (entry.transformEnabled ? "ON" : "OFF") << ","
             << entry.buildMode << ","
             << std::fixed << std::setprecision(2) << entry.fps << ","
             << entry.stdDev << "\n";
    }
    
    file.close();
    std::cout << "Performance data exported to: " << filename << std::endl;
    std::cout << "Total entries: " << entries.size() << std::endl;
}

void PerformanceLogger::clear() {
    entries.clear();
}

