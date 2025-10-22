#pragma once

#include <string>
#include <glad/gl.h>
#include <glm/glm.hpp>

class ShaderProgram {
public:
    ShaderProgram();
    ~ShaderProgram();
    
    bool loadFromFiles(const std::string& vertexPath, const std::string& fragmentPath);
    bool loadFromSource(const std::string& vertexSource, const std::string& fragmentSource);
    
    void use() const;
    void unuse() const;
    
    GLuint getProgram() const { return program; }
    
    // Uniform setters
    void setInt(const std::string& name, int value) const;
    void setFloat(const std::string& name, float value) const;
    void setVec2(const std::string& name, const glm::vec2& value) const;
    void setVec3(const std::string& name, const glm::vec3& value) const;
    void setVec4(const std::string& name, const glm::vec4& value) const;
    void setMat3(const std::string& name, const glm::mat3& value) const;
    void setMat4(const std::string& name, const glm::mat4& value) const;
    
private:
    GLuint program;
    
    bool compileShader(GLuint shader, const std::string& source);
    bool linkProgram();
    std::string readFile(const std::string& filepath);
};

