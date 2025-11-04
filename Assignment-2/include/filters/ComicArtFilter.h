#pragma once

#include "filters/Filter.h"

class CartoonFilter : public Filter {
public:
    CartoonFilter();
    
    void applyCPU(cv::Mat& frame) override;
    void applyGPU(ShaderProgram& shader) override;
    
    std::string getName() const override { return "Comic Art"; }
    FilterType getType() const override { return FilterType::Cartoon; }
    
    void setParameter(const std::string& name, float value) override;
    float getParameter(const std::string& name) const override;
    
private:
    int numDownSamples;
    int numBilateralFilters;
    float edgeThreshold;
};


