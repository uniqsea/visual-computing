#version 330 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D videoTexture;
uniform vec2 uvScale = vec2(1.0, 1.0);

void main() {
    vec2 uv = (TexCoord - 0.5) * uvScale + 0.5;
    FragColor = texture(videoTexture, uv);
}

