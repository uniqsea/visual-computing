#pragma once

#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl3.h>
#include <opencv2/opencv.hpp>

class Texture {
public:
    Texture();
    ~Texture();
    
    void create(int width, int height);
    void updateFromMat(const cv::Mat& mat, bool mirrorHorizontally = false);
    
    void bind(GLuint unit = 0) const;
    void unbind() const;
    
    GLuint getID() const { return textureID; }
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    
private:
    GLuint textureID;
    int width;
    int height;
};
