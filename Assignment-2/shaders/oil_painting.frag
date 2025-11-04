#version 330 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D videoTexture;
uniform int op_radius;     // 1..10
uniform int op_intensity;  // 4..64
uniform vec2 uvScale = vec2(1.0, 1.0);

// A lightweight approximation of oil painting:
// 1) Compute local luminance histogram (few bins)
// 2) Pick the bin with max frequency
// 3) Output the average color of that bin

void main()
{
    vec2 uv = (TexCoord - 0.5) * uvScale + 0.5;
    ivec2 texSize = textureSize(videoTexture, 0);
    vec2 texel = 1.0 / vec2(texSize);

    int radius = max(1, op_radius);
    int bins = clamp(op_intensity, 4, 64);

    // Accumulators
    const int MAX_BINS = 64;
    int counts[MAX_BINS];
    vec3 sums[MAX_BINS];
    for (int i = 0; i < MAX_BINS; ++i) {
        counts[i] = 0;
        sums[i] = vec3(0.0);
    }

    // Neighborhood
    for (int j = -radius; j <= radius; ++j) {
        for (int i = -radius; i <= radius; ++i) {
            vec3 c = texture(videoTexture, uv + vec2(i, j) * texel).rgb;
            float lum = dot(c, vec3(0.299, 0.587, 0.114));
            int idx = int(lum * float(bins));
            idx = clamp(idx, 0, bins - 1);
            counts[idx] += 1;
            sums[idx] += c;
        }
    }

    int best = 0;
    for (int k = 1; k < bins; ++k) {
        if (counts[k] > counts[best]) best = k;
    }

    vec3 color = texture(videoTexture, uv).rgb;
    if (counts[best] > 0) {
        color = sums[best] / float(counts[best]);
    }

    // Quantization strength linked to intensity for visible control
    // Map op_intensity in [4,64] -> q in [6,24]
    float t = clamp((float(op_intensity) - 4.0) / 60.0, 0.0, 1.0);
    float q = mix(6.0, 24.0, t);
    color = floor(color * q + 1e-5) / q;

    FragColor = vec4(color, 1.0);
}

