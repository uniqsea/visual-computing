#version 330 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D videoTexture;
uniform int blockSize = 10;

void main() {
    // Get texture dimensions
    vec2 texSize = vec2(textureSize(videoTexture, 0));
    
    // Calculate pixel size in texture coordinates
    vec2 pixelSize = vec2(blockSize) / texSize;
    
    // Calculate pixelated coordinates
    vec2 pixelatedCoord = floor(TexCoord / pixelSize) * pixelSize;
    
    // Add half pixel offset to sample from center of block
    pixelatedCoord += pixelSize * 0.5;
    
    // Sample texture
    FragColor = texture(videoTexture, pixelatedCoord);
}

