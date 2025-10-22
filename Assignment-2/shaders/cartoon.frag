#version 330 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D videoTexture;
uniform float edgeThreshold = 0.2;
uniform int quantizeLevels = 8;

// Sobel edge detection
float edgeDetection(vec2 texCoord) {
    vec2 texSize = vec2(textureSize(videoTexture, 0));
    vec2 texelSize = 1.0 / texSize;
    
    // Sobel kernels
    mat3 sobelX = mat3(
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0
    );
    
    mat3 sobelY = mat3(
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0,  1.0
    );
    
    float gx = 0.0;
    float gy = 0.0;
    
    // Sample 3x3 neighborhood
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            vec2 offset = vec2(float(i), float(j)) * texelSize;
            vec3 color = texture(videoTexture, texCoord + offset).rgb;
            float gray = dot(color, vec3(0.299, 0.587, 0.114));
            
            gx += gray * sobelX[i+1][j+1];
            gy += gray * sobelY[i+1][j+1];
        }
    }
    
    return sqrt(gx * gx + gy * gy);
}

// Color quantization
vec3 quantizeColor(vec3 color, int levels) {
    float step = 1.0 / float(levels);
    return floor(color / step) * step;
}

void main() {
    vec3 color = texture(videoTexture, TexCoord).rgb;
    
    // Detect edges
    float edge = edgeDetection(TexCoord);
    
    // Quantize colors for cartoon effect
    vec3 quantized = quantizeColor(color, quantizeLevels);
    
    // Apply edge darkening
    if (edge > edgeThreshold) {
        quantized *= 0.0; // Make edges black
    }
    
    FragColor = vec4(quantized, 1.0);
}

