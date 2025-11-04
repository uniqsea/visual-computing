#include "utils/PerformanceLogger.h"
#include <iostream>
#include <iomanip>
#include <filesystem>

PerformanceLogger::PerformanceLogger() {
}

void PerformanceLogger::addEntry(const PerformanceData& data) {
    entries.push_back(data);
    std::cout << "Logged: " << data.resolution << " | " << data.filter 
              << " | " << data.mode << " | Transform: " << (data.transformEnabled ? "ON" : "OFF")
              << " | " << data.buildMode << " | FrameAvgMs: " << std::fixed 
              << std::setprecision(2) << data.frameTimeAvgMs 
              << ", AlgoAvgMs: " << data.algoTimeAvgMs << std::endl;
}

bool PerformanceLogger::exportToCSV(const std::string& filename) {
    namespace fs = std::filesystem;
    auto tryWrite = [&](const std::string& path) -> bool {
        try {
            fs::path p(path);
            if (p.has_parent_path()) {
                fs::create_directories(p.parent_path());
            }

            // Determine whether to write header (new file or empty file)
            bool writeHeader = true;
            if (fs::exists(p)) {
                std::error_code ec;
                auto sz = fs::file_size(p, ec);
                if (!ec && sz > 0) writeHeader = false;
            }

            // Append mode to avoid overwriting previous runs
            std::ofstream file(path, std::ios::app);
            if (!file.is_open()) {
                return false;
            }

            if (writeHeader) {
                file << "Resolution,Filter,Mode,Transform,BuildMode,FrameTimeAvgMs,AlgoTimeAvgMs,FrameTimeFPS,AlgoTimeFPS,SampleCount\n";
            }
            // Write current session entries then clear to avoid duplicates on next export
            for (const auto& entry : entries) {
                const double frameFps = (entry.frameTimeAvgMs > 0.0f) ? (1000.0 / static_cast<double>(entry.frameTimeAvgMs)) : 0.0;
                const double algoFps  = (entry.algoTimeAvgMs  > 0.0f) ? (1000.0 / static_cast<double>(entry.algoTimeAvgMs))  : 0.0;
                file << entry.resolution << ","
                     << entry.filter << ","
                     << entry.mode << ","
                     << (entry.transformEnabled ? "ON" : "OFF") << ","
                     << entry.buildMode << ","
                     << std::fixed << std::setprecision(2)
                     << entry.frameTimeAvgMs << ","
                     << entry.algoTimeAvgMs << ","
                     << frameFps << ","
                     << algoFps << ","
                     << entry.sampleCount << "\n";
            }
            file.close();

            std::cout << "Performance data exported to: " << path << std::endl;
            std::cout << "Appended entries: " << entries.size() << std::endl;

            // Avoid duplicating rows on subsequent exports in the same session
            entries.clear();
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Export error for '" << path << "': " << e.what() << std::endl;
            return false;
        }
    };

    // Try default path first (relative to current working dir)
    if (tryWrite(filename)) {
        return true;
    }
    // If default fails and looks like "data/...", try writing to ../data for typical build/ run
    if (filename.rfind("data/", 0) == 0) {
        std::string fallback = std::string("../") + filename;
        if (tryWrite(fallback)) {
            return true;
        }
    }
    std::cerr << "Failed to export performance data to '" << filename << "' and fallback." << std::endl;
    return false;
}

void PerformanceLogger::clear() {
    entries.clear();
}
