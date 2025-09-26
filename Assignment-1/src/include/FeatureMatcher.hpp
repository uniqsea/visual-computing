#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <chrono>

enum class MatcherType {
    BRUTE_FORCE,
    FLANN
};

struct MatchingResult {
    std::vector<cv::DMatch> matches;
    std::vector<cv::DMatch> good_matches;
    double matching_time_ms;
    int total_matches;
    int good_matches_count;
    double match_ratio;
    std::vector<double> match_distances;
    MatcherType matcher_type;
};

class FeatureMatcher {
private:
    MatcherType matcher_type_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
    double ratio_threshold_;

public:
    FeatureMatcher(MatcherType type, double ratio_threshold = 0.75);
    ~FeatureMatcher() = default;

    MatchingResult matchFeatures(const cv::Mat& descriptors1, const cv::Mat& descriptors2);

    void setRatioThreshold(double ratio) { ratio_threshold_ = ratio; }
    double getRatioThreshold() const { return ratio_threshold_; }

    static std::string matcherTypeToString(MatcherType type);

private:
    void initializeMatcher(const cv::Mat& descriptors1, const cv::Mat& descriptors2);
    std::vector<cv::DMatch> filterMatchesWithRatio(const std::vector<std::vector<cv::DMatch>>& knn_matches);
    std::vector<cv::DMatch> filterMatchesWithDistance(const std::vector<cv::DMatch>& matches);
};