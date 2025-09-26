#include "FeatureDetector.hpp"
#include <iostream>
#include <chrono>

using namespace std;

FeatureDetector::FeatureDetector(DetectorType type) : detector_type_(type) {
    setDetectorParameters(type);
}

void FeatureDetector::setDetectorParameters(DetectorType type, const map<string, double>&) {
    detector_type_ = type;

    switch (type) {
        case DetectorType::SIFT:
            detector_ = cv::SIFT::create();
            break;
        case DetectorType::ORB:
            detector_ = cv::ORB::create(10000); // Default 100000 features
            break;
        case DetectorType::AKAZE:
            detector_ = cv::AKAZE::create();
            break;
    }
}

DetectionResult FeatureDetector::detectAndCompute(const cv::Mat& image) {
    DetectionResult result;
    result.detector_type = detector_type_;

    auto start_time = chrono::high_resolution_clock::now();

    // Ensure grayscale for stable keypoints/descriptors across detectors
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = image;
    }

    detector_->detectAndCompute(gray, cv::Mat(), result.keypoints, result.descriptors);

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
    result.detection_time_ms = duration.count() / 1000.0;

    result.keypoint_count = static_cast<int>(result.keypoints.size());

    return result;
}

string FeatureDetector::detectorTypeToString(DetectorType type) {
    switch (type) {
        case DetectorType::SIFT: return "SIFT";
        case DetectorType::ORB: return "ORB";
        case DetectorType::AKAZE: return "AKAZE";
        default: return "Unknown";
    }
}
