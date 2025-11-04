#include "filters/PixelationFilter.h"
#include "ShaderProgram.h"

PixelationFilter::PixelationFilter(int blockSize) : blockSize(blockSize) {
}

void PixelationFilter::applyCPU(cv::Mat& frame) {
    if (frame.empty() || blockSize <= 1) {
        return;
    }
    
    int originalWidth = frame.cols;
    int originalHeight = frame.rows;
    
    // Calculate downscaled size
    int smallWidth = originalWidth / blockSize;
    int smallHeight = originalHeight / blockSize;
    
    if (smallWidth < 1) smallWidth = 1;
    if (smallHeight < 1) smallHeight = 1;
    
    cv::Mat small;
    
    // Downscale (INTER_AREA is better for shrinking and reduces aliasing)
    cv::resize(frame, small, cv::Size(smallWidth, smallHeight), 0, 0, cv::INTER_AREA);
    
    // Upscale back to original size with nearest neighbor interpolation
    cv::resize(small, frame, cv::Size(originalWidth, originalHeight), 0, 0, cv::INTER_NEAREST);
}

void PixelationFilter::applyGPU(ShaderProgram& shader) {
    shader.setInt("blockSize", blockSize);
}

void PixelationFilter::setParameter(const std::string& name, float value) {
    if (name == "blockSize") {
        blockSize = static_cast<int>(value);
        if (blockSize < 1) blockSize = 1;
        if (blockSize > 100) blockSize = 100;
    }
}

float PixelationFilter::getParameter(const std::string& name) const {
    if (name == "blockSize") {
        return static_cast<float>(blockSize);
    }
    return 0.0f;
}

