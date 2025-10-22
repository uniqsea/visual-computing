#include "CartoonFilter.h"
#include "../ShaderProgram.h"

CartoonFilter::CartoonFilter() 
    : numDownSamples(2), numBilateralFilters(7), edgeThreshold(50.0f) {
}

void CartoonFilter::applyCPU(cv::Mat& frame) {
    if (frame.empty()) {
        return;
    }
    
    cv::Mat imgColor;
    frame.copyTo(imgColor);
    
    // Downsample image using Gaussian pyramid
    for (int i = 0; i < numDownSamples; i++) {
        cv::pyrDown(imgColor, imgColor);
    }
    
    // Apply bilateral filter to reduce color palette
    for (int i = 0; i < numBilateralFilters; i++) {
        cv::bilateralFilter(imgColor.clone(), imgColor, 9, 9, 7);
    }
    
    // Upsample image to original size
    for (int i = 0; i < numDownSamples; i++) {
        cv::pyrUp(imgColor, imgColor);
    }
    
    // Resize to exact original size if needed
    if (imgColor.size() != frame.size()) {
        cv::resize(imgColor, imgColor, frame.size());
    }
    
    // Edge detection
    cv::Mat imgGray;
    cv::cvtColor(frame, imgGray, cv::COLOR_BGR2GRAY);
    
    // Median blur to reduce noise
    cv::medianBlur(imgGray, imgGray, 7);
    
    // Adaptive threshold for edges
    cv::Mat imgEdge;
    cv::adaptiveThreshold(imgGray, imgEdge, 255, 
                         cv::ADAPTIVE_THRESH_MEAN_C, 
                         cv::THRESH_BINARY, 9, 2);
    
    // Convert edge image to color
    cv::Mat imgEdgeColor;
    cv::cvtColor(imgEdge, imgEdgeColor, cv::COLOR_GRAY2BGR);
    
    // Combine color and edge images
    cv::bitwise_and(imgColor, imgEdgeColor, frame);
}

void CartoonFilter::applyGPU(ShaderProgram& shader) {
    shader.setFloat("edgeThreshold", edgeThreshold / 255.0f);
    shader.setInt("quantizeLevels", 8);
}

void CartoonFilter::setParameter(const std::string& name, float value) {
    if (name == "edgeThreshold") {
        edgeThreshold = value;
    } else if (name == "numBilateralFilters") {
        numBilateralFilters = static_cast<int>(value);
    }
}

float CartoonFilter::getParameter(const std::string& name) const {
    if (name == "edgeThreshold") {
        return edgeThreshold;
    } else if (name == "numBilateralFilters") {
        return static_cast<float>(numBilateralFilters);
    }
    return 0.0f;
}

