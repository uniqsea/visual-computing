#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <string>
#include <map>

enum class DetectorType {
    SIFT,
    ORB,
    AKAZE
};

struct DetectionResult {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    double detection_time_ms;
    int keypoint_count;
    DetectorType detector_type;
};

class FeatureDetector {
private:
    DetectorType detector_type_;
    cv::Ptr<cv::Feature2D> detector_;

public:
    FeatureDetector(DetectorType type);
    ~FeatureDetector() = default;

    DetectionResult detectAndCompute(const cv::Mat& image);

    void setDetectorParameters(DetectorType type, const std::map<std::string, double>& params = {});

    static std::string detectorTypeToString(DetectorType type);

    DetectorType getDetectorType() const { return detector_type_; }
};
