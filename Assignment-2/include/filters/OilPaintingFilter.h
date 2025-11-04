#pragma once

#include "filters/Filter.h"

class OilPaintingFilter : public Filter {
public:
    OilPaintingFilter();
    
    void applyCPU(cv::Mat& frame) override;
    void applyGPU(ShaderProgram& shader) override;
    
    std::string getName() const override { return "OilPainting"; }
    FilterType getType() const override { return FilterType::OilPainting; }
    
    void setParameter(const std::string& name, float value) override;
    float getParameter(const std::string& name) const override;
    
private:
    int radius;
    int intensity;
};

