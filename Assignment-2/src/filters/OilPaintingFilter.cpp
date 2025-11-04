#include "filters/OilPaintingFilter.h"
#include "ShaderProgram.h"
#include <vector>
#include <algorithm>
#include <cmath>
// Unified semantics with a single exposed parameter: Brush Radius

OilPaintingFilter::OilPaintingFilter() : radius(6), intensity(24) {}

void OilPaintingFilter::applyCPU(cv::Mat& frame) {
    if (frame.empty()) return;
    
    // CPU path: histogram bins fixed; only brush radius controls neighborhood size
    cv::Mat src;
    frame.copyTo(src);
    cv::Mat result = src.clone();
    const int bins = 24; // fixed bins for stable look matching GPU default
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            int size = std::max(1, radius);
            int x0 = std::max(0, x - size);
            int x1 = std::min(src.cols - 1, x + size);
            int y0 = std::max(0, y - size);
            int y1 = std::min(src.rows - 1, y + size);
            std::vector<int> count(bins, 0);
            std::vector<cv::Vec3i> sum(bins, cv::Vec3i(0, 0, 0));
            for (int j = y0; j <= y1; ++j) {
                const cv::Vec3b* row = src.ptr<cv::Vec3b>(j);
                for (int i = x0; i <= x1; ++i) {
                    const cv::Vec3b& c = row[i];
                    int idx = ((int)c[0] + (int)c[1] + (int)c[2]) * bins / (256 * 3);
                    idx = std::min(bins - 1, std::max(0, idx));
                    count[idx]++;
                    sum[idx][0] += c[0];
                    sum[idx][1] += c[1];
                    sum[idx][2] += c[2];
                }
            }
            int best = 0;
            for (int k = 1; k < bins; ++k) {
                if (count[k] > count[best]) best = k;
            }
            if (count[best] > 0) {
                float b = (float)sum[best][0] / (float)count[best];
                float g = (float)sum[best][1] / (float)count[best];
                float r = (float)sum[best][2] / (float)count[best];
                // Fixed quantization strength to match GPU constant intensity (q≈12)
                float q = 12.0f;
                auto quant = [q](float v) -> uchar {
                    float nv = std::floor((v / 255.0f) * q + 1e-5f) / q;
                    nv = std::min(1.0f, std::max(0.0f, nv));
                    return (uchar)std::round(nv * 255.0f);
                };
                result.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    quant(b), quant(g), quant(r)
                );
            }
        }
    }
    frame = result;
}

void OilPaintingFilter::applyGPU(ShaderProgram& shader) {
    shader.setInt("op_radius", radius);
    shader.setInt("op_intensity", 24); // fixed intensity to match CPU bins/quantization
}

void OilPaintingFilter::setParameter(const std::string& name, float value) {
    // Single-parameter design: only brush radius is adjustable
    if (name == "radius") radius = std::max(1, (int)value);
}

float OilPaintingFilter::getParameter(const std::string& name) const {
    if (name == "radius") return (float)radius;
    return 0.0f;
}
