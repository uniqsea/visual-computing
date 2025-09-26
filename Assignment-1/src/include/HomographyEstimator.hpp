#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

struct HomographyResult {
    cv::Mat homography;
    std::vector<uchar> inliers_mask;
    int inliers_count;
    double estimation_time_ms;
    double reprojection_threshold;
    double confidence;
    bool valid;
};

class HomographyEstimator {
private:
    double reprojection_threshold_;
    double confidence_;
    int max_iterations_;

public:
    HomographyEstimator(double reprojection_threshold = 3.0,
                       double confidence = 0.99,
                       int max_iterations = 2000);
    ~HomographyEstimator() = default;

    HomographyResult estimateHomography(
        const std::vector<cv::Point2f>& points1,
        const std::vector<cv::Point2f>& points2);

    HomographyResult estimateHomographyFromMatches(
        const std::vector<cv::KeyPoint>& keypoints1,
        const std::vector<cv::KeyPoint>& keypoints2,
        const std::vector<cv::DMatch>& matches);

    void setReprojectionThreshold(double threshold) { reprojection_threshold_ = threshold; }
    double getReprojectionThreshold() const { return reprojection_threshold_; }

    void setConfidence(double confidence) { confidence_ = confidence; }
    double getConfidence() const { return confidence_; }

    void setMaxIterations(int max_iter) { max_iterations_ = max_iter; }
    int getMaxIterations() const { return max_iterations_; }

private:
    std::vector<cv::Point2f> extractPoints(
        const std::vector<cv::KeyPoint>& keypoints,
        const std::vector<cv::DMatch>& matches,
        bool query_points);
};
