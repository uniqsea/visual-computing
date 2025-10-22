#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

uniform mat4 transform = mat4(1.0);

void main() {
    // Apply transform if provided
    vec4 transformedPos = transform * vec4(aPos, 1.0);
    gl_Position = transformedPos;
    
    TexCoord = aTexCoord;
}

