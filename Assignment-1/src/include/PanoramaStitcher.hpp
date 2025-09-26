#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

enum class BlendingType {
    SIMPLE_OVERLAY,
    FEATHERING,
    MULTIBAND
};

struct StitchingResult {
    cv::Mat panorama;
    double stitching_time_ms;
    BlendingType blending_type;
    cv::Size output_size;
    bool success;
};

class PanoramaStitcher {
private:
    BlendingType blending_type_;

public:
    PanoramaStitcher(BlendingType blending_type = BlendingType::FEATHERING);
    ~PanoramaStitcher() = default;

    StitchingResult stitchImages(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& homography);

    // Global warping + single-pass blending given transforms to a common reference
    StitchingResult stitchImagesGlobal(const std::vector<cv::Mat>& images,
                                       const std::vector<cv::Mat>& transforms_to_ref);

    void setBlendingType(BlendingType type) { blending_type_ = type; }
    BlendingType getBlendingType() const { return blending_type_; }

    static std::string blendingTypeToString(BlendingType type);

private:
    cv::Mat simpleOverlayBlend(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& homography);

    cv::Mat featheringBlend(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& homography);

    cv::Mat multibandBlend(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& homography);

    std::vector<cv::Point2f> getImageCorners(const cv::Mat& image);

    cv::Rect getBoundingRect(const std::vector<cv::Point2f>& points);

    // Helpers for global stitching
    std::vector<cv::Point2f> getTransformedCorners(const cv::Mat& image, const cv::Mat& H);
    cv::Rect computeGlobalBoundingRect(const std::vector<cv::Mat>& images,
                                       const std::vector<cv::Mat>& transforms);
};
