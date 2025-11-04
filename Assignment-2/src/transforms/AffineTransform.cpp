#include "transforms/AffineTransform.h"
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

AffineTransform::AffineTransform() 
    : translation(0.0f, 0.0f), rotation(0.0f), scaleValue(1.0f) {
}

void AffineTransform::reset() {
    translation = glm::vec2(0.0f, 0.0f);
    rotation = 0.0f;
    scaleValue = 1.0f;
}

void AffineTransform::translate(float dx, float dy) {
    translation.x += dx;
    translation.y += dy;
}

void AffineTransform::rotate(float angle) {
    rotation += angle;
}

void AffineTransform::scale(float factor) {
    scaleValue *= factor;
    if (scaleValue < 0.1f) scaleValue = 0.1f;
    if (scaleValue > 5.0f) scaleValue = 5.0f;
}

void AffineTransform::setTranslation(float x, float y) {
    translation = glm::vec2(x, y);
}

void AffineTransform::setRotation(float angle) {
    rotation = angle;
}

void AffineTransform::setScale(float s) {
    scaleValue = s;
    if (scaleValue < 0.1f) scaleValue = 0.1f;
    if (scaleValue > 5.0f) scaleValue = 5.0f;
}

cv::Mat AffineTransform::getOpenCVMatrix(int frameWidth, int frameHeight) const {
    // Create transformation matrix for OpenCV
    // Center of the image
    cv::Point2f center(frameWidth / 2.0f, frameHeight / 2.0f);
    
    // Get rotation matrix around center
    cv::Mat rotMat = cv::getRotationMatrix2D(center, rotation * 180.0f / M_PI, scaleValue);
    
    // Add translation
    // Note: final texture upload flips vertically for OpenGL coordinates.
    // Invert Y here so on-screen panning matches GPU behavior.
    rotMat.at<double>(0, 2) += translation.x;
    rotMat.at<double>(1, 2) += -translation.y;
    
    return rotMat;
}

void AffineTransform::applyCPU(cv::Mat& frame) {
    if (frame.empty()) {
        return;
    }
    
    // Skip if identity transform
    if (translation.x == 0.0f && translation.y == 0.0f && 
        rotation == 0.0f && scaleValue == 1.0f) {
        return;
    }
    
    cv::Mat transformMatrix = getOpenCVMatrix(frame.cols, frame.rows);
    cv::Mat result;
    
    cv::warpAffine(frame, result, transformMatrix, frame.size(), 
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    
    frame = result;
}

glm::mat4 AffineTransform::getGPUMatrix() const {
    glm::mat4 matrix = glm::mat4(1.0f);
    
    // Apply transformations in order: scale, rotate, translate
    matrix = glm::translate(matrix, glm::vec3(translation.x, translation.y, 0.0f));
    matrix = glm::rotate(matrix, rotation, glm::vec3(0.0f, 0.0f, 1.0f));
    matrix = glm::scale(matrix, glm::vec3(scaleValue, scaleValue, 1.0f));
    
    return matrix;
}

glm::mat3 AffineTransform::getGPUMatrix3() const {
    glm::mat3 matrix = glm::mat3(1.0f);
    
    // Scale
    matrix[0][0] = scaleValue;
    matrix[1][1] = scaleValue;
    
    // Rotation
    float cosR = std::cos(rotation);
    float sinR = std::sin(rotation);
    
    glm::mat3 rotMat = glm::mat3(1.0f);
    rotMat[0][0] = cosR;
    rotMat[0][1] = sinR;
    rotMat[1][0] = -sinR;
    rotMat[1][1] = cosR;
    
    matrix = rotMat * matrix;
    
    // Translation (in homogeneous coordinates)
    matrix[2][0] = translation.x;
    matrix[2][1] = translation.y;
    
    return matrix;
}

glm::mat4 AffineTransform::getGPUMatrixForViewport(float viewportWidth, float viewportHeight) const {
    glm::mat4 matrix = glm::mat4(1.0f);
    float ndcX = (viewportWidth  > 0.0f) ? (2.0f * translation.x / viewportWidth) : 0.0f;
    float ndcY = (viewportHeight > 0.0f) ? (-2.0f * translation.y / viewportHeight) : 0.0f; // invert y for screen coords
    matrix = glm::translate(matrix, glm::vec3(ndcX, ndcY, 0.0f));
    matrix = glm::rotate(matrix, rotation, glm::vec3(0.0f, 0.0f, 1.0f));
    matrix = glm::scale(matrix, glm::vec3(scaleValue, scaleValue, 1.0f));
    return matrix;
}
