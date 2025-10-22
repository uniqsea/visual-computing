#pragma once

#include <opencv2/opencv.hpp>
#include <glm/glm.hpp>

class Transform {
public:
    virtual ~Transform() = default;
    
    virtual void applyCPU(cv::Mat& frame) = 0;
    virtual glm::mat4 getGPUMatrix() const = 0;
    virtual glm::mat3 getGPUMatrix3() const = 0;
    
    virtual void reset() = 0;
};

