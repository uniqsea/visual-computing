#pragma once

#include <opencv2/opencv.hpp>
#include <string>

class ShaderProgram;

enum class FilterType {
    None,
    Pixelation,
    Cartoon,
    OilPainting
};

class Filter {
public:
    virtual ~Filter() = default;
    
    virtual void applyCPU(cv::Mat& frame) = 0;
    virtual void applyGPU(ShaderProgram& shader) = 0;
    
    virtual std::string getName() const = 0;
    virtual FilterType getType() const = 0;
    
    // Parameter adjustment (optional)
    virtual void setParameter(const std::string& name, float value) {}
    virtual float getParameter(const std::string& name) const { return 0.0f; }
};

