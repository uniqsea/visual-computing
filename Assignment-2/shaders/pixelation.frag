#version 330 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D videoTexture;
uniform int blockSize = 10;
uniform vec2 uvScale = vec2(1.0, 1.0);

void main() {
    // Texture size (in texels)
    vec2 texSize = vec2(textureSize(videoTexture, 0));
    
    // Apply cover scaling first
    vec2 uv = (TexCoord - 0.5) * uvScale + 0.5;
    
    // Quantize by integer texels per block to match CPU alignment
    float bs = float(blockSize);
    vec2 pix = uv * texSize;
    vec2 blockIdx = floor(pix / bs);
    vec2 centerPix = (blockIdx + 0.5) * bs; // center of the block in texel space
    vec2 sampleUV = centerPix / texSize;
    
    // Use an effective scale-based LOD to better match non-divisible sizes
    vec2 blocks = max(floor(texSize / bs), vec2(1.0));
    vec2 scaleEff = texSize / blocks; // effective texels per block
    float lod = max(log2(max(scaleEff.x, scaleEff.y)), 0.0);
    
    FragColor = textureLod(videoTexture, sampleUV, lod);
}
