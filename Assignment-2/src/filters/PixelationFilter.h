#pragma once

#include "Filter.h"

class PixelationFilter : public Filter {
public:
    PixelationFilter(int blockSize = 10);
    
    void applyCPU(cv::Mat& frame) override;
    void applyGPU(ShaderProgram& shader) override;
    
    std::string getName() const override { return "Pixelation"; }
    FilterType getType() const override { return FilterType::Pixelation; }
    
    void setParameter(const std::string& name, float value) override;
    float getParameter(const std::string& name) const override;
    
    void setBlockSize(int size) { blockSize = size; }
    int getBlockSize() const { return blockSize; }
    
private:
    int blockSize;
};

