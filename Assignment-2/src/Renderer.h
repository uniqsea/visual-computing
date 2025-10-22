#pragma once

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include "ShaderProgram.h"
#include "Texture.h"
#include <memory>

class Renderer {
public:
    Renderer();
    ~Renderer();
    
    bool initialize(int width, int height);
    void shutdown();
    
    void clear(float r = 0.0f, float g = 0.0f, float b = 0.0f, float a = 1.0f);
    void renderTexturedQuad(const Texture& texture, ShaderProgram& shader);
    
    int getWidth() const { return windowWidth; }
    int getHeight() const { return windowHeight; }
    
private:
    int windowWidth;
    int windowHeight;
    
    GLuint quadVAO;
    GLuint quadVBO;
    
    void setupQuad();
};

