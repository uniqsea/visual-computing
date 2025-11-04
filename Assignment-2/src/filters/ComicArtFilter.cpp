#include "filters/ComicArtFilter.h"
#include "ShaderProgram.h"
#include <cmath>

CartoonFilter::CartoonFilter() 
    : numDownSamples(2), numBilateralFilters(7), edgeThreshold(55.0f) {
}

void CartoonFilter::applyCPU(cv::Mat& frame) {
    if (frame.empty()) {
        return;
    }
    
    // 1) Color quantization (match GPU quantizeColor)
    cv::Mat imgColor;
    frame.copyTo(imgColor);
    cv::Mat colorF;
    imgColor.convertTo(colorF, CV_32F, 1.0 / 255.0);
    const int levels = 8; // match GPU quantizeLevels
    // quantized = floor(colorF * levels) / levels; (values are non-negative)
    cv::Mat tmpMul = colorF * static_cast<float>(levels);
    cv::Mat tmpInt; tmpMul.convertTo(tmpInt, CV_32S);
    cv::Mat tmpF; tmpInt.convertTo(tmpF, CV_32F);
    cv::Mat quantized = tmpF / static_cast<float>(levels);
    quantized.convertTo(imgColor, CV_8U, 255.0);
    
    // 2) Edge detection with Sobel magnitude, threshold controlled by edgeThreshold
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Mat grayf;
    gray.convertTo(grayf, CV_32F, 1.0 / 255.0);
    
    cv::Mat gx, gy;
    // 3x3 Sobel (lightweight, matches GPU edgeDetection)
    cv::Sobel(grayf, gx, CV_32F, 1, 0, 3);
    cv::Sobel(grayf, gy, CV_32F, 0, 1, 3);
    cv::Mat mag;
    cv::magnitude(gx, gy, mag);
    // Normalize by theoretical max Sobel response for inputs in [0,1]
    const float SOBEL_MAX = 4.0f * 1.41421356f; // 4*sqrt(2)
    mag /= SOBEL_MAX;
    
    // Threshold with gamma>1 to increase low-end sensitivity (more edges for small values)
    float thr = std::pow(edgeThreshold / 255.0f, 1.6f);
    cv::Mat edgeMask;
    cv::threshold(mag, edgeMask, thr, 1.0, cv::THRESH_BINARY);
    edgeMask.convertTo(edgeMask, CV_8U, 255.0);
    // No morphology: keep edges thin and cost low (GPU mirrors this)
    
    // 3) Darken edges (set to black where mask is 1), like GPU
    cv::Mat result = imgColor.clone();
    result.setTo(cv::Scalar(0, 0, 0), edgeMask);
    frame = result;
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
