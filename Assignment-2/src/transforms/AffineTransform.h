#pragma once

#include "Transform.h"

class AffineTransform : public Transform {
public:
    AffineTransform();
    
    void applyCPU(cv::Mat& frame) override;
    glm::mat4 getGPUMatrix() const override;
    glm::mat3 getGPUMatrix3() const override;
    
    void reset() override;
    
    // Transformation controls
    void translate(float dx, float dy);
    void rotate(float angle); // angle in radians
    void scale(float factor);
    
    void setTranslation(float x, float y);
    void setRotation(float angle);
    void setScale(float s);
    
    // Getters
    glm::vec2 getTranslation() const { return translation; }
    float getRotation() const { return rotation; }
    float getScale() const { return scaleValue; }
    
private:
    glm::vec2 translation;  // Translation offset
    float rotation;         // Rotation angle in radians
    float scaleValue;       // Scale factor
    
    cv::Mat getOpenCVMatrix(int frameWidth, int frameHeight) const;
};

