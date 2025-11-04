#include "Texture.h"
#include <iostream>

Texture::Texture() : textureID(0), width(0), height(0) {
}

Texture::~Texture() {
    if (textureID != 0) {
        glDeleteTextures(1, &textureID);
    }
}

void Texture::create(int w, int h) {
    width = w;
    height = h;
    
    if (textureID != 0) {
        glDeleteTextures(1, &textureID);
    }
    
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    
    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // Use mipmaps so GPU can approximate area averaging
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    // Allocate texture memory
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glGenerateMipmap(GL_TEXTURE_2D);
    
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Texture::updateFromMat(const cv::Mat& mat, bool mirrorHorizontally) {
    if (mat.empty()) {
        std::cerr << "Cannot update texture from empty Mat" << std::endl;
        return;
    }
    
    // OpenCV stores top-left origin (BGR). Convert to RGB and flip for OpenGL coords;
    // optionally mirror horizontally for a selfie-style preview.
    cv::Mat rgb;
    cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    if (mirrorHorizontally) {
        // Mirror horizontally only (selfie view), keep vertical flip for OpenGL coords handled below
        // First flip vertically to convert from OpenCV top-left to OpenGL bottom-left
        cv::flip(rgb, rgb, 0);
        // Then mirror horizontally
        cv::flip(rgb, rgb, 1);
    } else {
        // Only vertical flip to correct coordinate origin
        cv::flip(rgb, rgb, 0);
    }
    
    // Update texture size if needed
    if (width != rgb.cols || height != rgb.rows) {
        create(rgb.cols, rgb.rows);
    }
    
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Texture::bind(GLuint unit) const {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, textureID);
}

void Texture::unbind() const {
    glBindTexture(GL_TEXTURE_2D, 0);
}
