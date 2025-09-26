#pragma once

#include "FeatureDetector.hpp"
#include "FeatureMatcher.hpp"
#include "HomographyEstimator.hpp"
#include "PanoramaStitcher.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <string>
#include <fstream>

struct ExperimentResult {
    DetectorType detector_type;
    MatcherType matcher_type;
    BlendingType blending_type;
    double reprojection_threshold;

    int keypoints_img1;
    int keypoints_img2;
    double detection_time_ms;

    int total_matches;
    int good_matches;
    double match_ratio;
    double matching_time_ms;

    int inliers_count;
    double homography_time_ms;

    double stitching_time_ms;
    cv::Size output_size;
    bool stitching_success;

    double total_time_ms;
};

struct ExperimentConfig {
    std::vector<DetectorType> detector_types;
    std::vector<MatcherType> matcher_types;
    std::vector<BlendingType> blending_types;
    std::vector<double> reprojection_thresholds;
    std::string output_dir;
    bool save_intermediate_results;
    bool save_visualizations;
};

class Evaluator {
private:
    ExperimentConfig config_;
    std::vector<ExperimentResult> results_;

public:
    Evaluator(const ExperimentConfig& config);
    ~Evaluator() = default;

    void runExperiments(const cv::Mat& img1, const cv::Mat& img2, const std::string& experiment_name = "");

    void runBatchExperiments(const std::vector<std::pair<cv::Mat, cv::Mat>>& image_pairs,
                            const std::vector<std::string>& experiment_names = {});

    void saveResults(const std::string& filename) const;

    void saveResultsAsTable(const std::string& filename) const;

    void generateReport(const std::string& filename) const;

    void plotMatchDistanceHistogram(const std::vector<double>& distances,
                                  const std::string& filename,
                                  const std::string& title = "") const;

    void saveVisualization(const cv::Mat& img1, const cv::Mat& img2,
                          const std::vector<cv::KeyPoint>& keypoints1,
                          const std::vector<cv::KeyPoint>& keypoints2,
                          const std::vector<cv::DMatch>& matches,
                          const std::string& filename) const;

    // Save keypoints as crosses (X) for a single image
    void saveKeypointsCross(const cv::Mat& img,
                            const std::vector<cv::KeyPoint>& keypoints,
                            const std::string& filename,
                            const cv::Scalar& color = cv::Scalar(0, 255, 0),
                            int thickness = 1) const;

    void clearResults() { results_.clear(); }

    const std::vector<ExperimentResult>& getResults() const { return results_; }

    void setConfig(const ExperimentConfig& config) { config_ = config; }

private:
    std::string formatDuration(double ms) const;

    void ensureOutputDirectory() const;

    void saveIntermediateImage(const cv::Mat& image, const std::string& filename) const;
};
