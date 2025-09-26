#include "Evaluator.hpp"
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <algorithm>

using namespace std;

Evaluator::Evaluator(const ExperimentConfig& config) : config_(config) {
    ensureOutputDirectory();
}

void Evaluator::runExperiments(const cv::Mat& img1, const cv::Mat& img2, const string& experiment_name) {
    cout << "Running experiments" << (experiment_name.empty() ? "" : " for " + experiment_name) << "..." << endl;

    for (auto detector_type : config_.detector_types) {
        for (auto matcher_type : config_.matcher_types) {
            for (auto blending_type : config_.blending_types) {
                for (double threshold : config_.reprojection_thresholds) {
                    ExperimentResult result;
                    result.detector_type = detector_type;
                    result.matcher_type = matcher_type;
                    result.blending_type = blending_type;
                    result.reprojection_threshold = threshold;

                    auto start_total = chrono::high_resolution_clock::now();

                    FeatureDetector detector(detector_type);

                    auto detection1 = detector.detectAndCompute(img1);
                    auto detection2 = detector.detectAndCompute(img2);

                    result.keypoints_img1 = detection1.keypoint_count;
                    result.keypoints_img2 = detection2.keypoint_count;
                    result.detection_time_ms = detection1.detection_time_ms + detection2.detection_time_ms;

                    FeatureMatcher matcher(matcher_type);
                    auto matching_result = matcher.matchFeatures(detection1.descriptors, detection2.descriptors);

                    result.total_matches = matching_result.total_matches;
                    result.good_matches = matching_result.good_matches_count;
                    result.match_ratio = matching_result.match_ratio;
                    result.matching_time_ms = matching_result.matching_time_ms;

                    HomographyEstimator estimator(threshold);
                    auto homography_result = estimator.estimateHomographyFromMatches(
                        detection1.keypoints, detection2.keypoints, matching_result.good_matches);

                    result.inliers_count = homography_result.inliers_count;
                    result.homography_time_ms = homography_result.estimation_time_ms;

                    if (homography_result.valid) {
                        PanoramaStitcher stitcher(blending_type);
                        // Use H mapping img2 -> img1 for stitching
                        cv::Mat H21 = homography_result.homography.inv();
                        auto stitching_result = stitcher.stitchImages(img1, img2, H21);

                        result.stitching_time_ms = stitching_result.stitching_time_ms;
                        result.output_size = stitching_result.output_size;
                        result.stitching_success = stitching_result.success;

                        // Prepare hierarchical output directory per experiment/config
                        const std::string detStr = FeatureDetector::detectorTypeToString(detector_type);
                        const std::string matStr = FeatureMatcher::matcherTypeToString(matcher_type);
                        const std::string blendStr = PanoramaStitcher::blendingTypeToString(blending_type);
                        const std::string thrStr = std::string("t") + std::to_string(static_cast<int>(threshold * 10));
                        const std::string expName = experiment_name.empty() ? std::string("experiment") : experiment_name;
                        const std::string baseDir = config_.output_dir + "/" + expName + "/" + detStr + "/" + matStr + "/" + blendStr + "/" + thrStr;
                        std::filesystem::create_directories(baseDir);

                        if (config_.save_intermediate_results && stitching_result.success) {
                            string filename = baseDir + "/stitch.jpg";
                            saveIntermediateImage(stitching_result.panorama, filename);
                        }

                        if (config_.save_visualizations) {
                            // Matches visualization
                            string vis_filename = baseDir + "/matches.jpg";
                            saveVisualization(img1, img2, detection1.keypoints, detection2.keypoints,
                                              matching_result.good_matches, vis_filename);

                            // Keypoints as crosses (X) for both images
                            string kps1_filename = baseDir + "/kps_img1.jpg";
                            string kps2_filename = baseDir + "/kps_img2.jpg";
                            saveKeypointsCross(img1, detection1.keypoints, kps1_filename, cv::Scalar(0,255,0), 1);
                            saveKeypointsCross(img2, detection2.keypoints, kps2_filename, cv::Scalar(0,255,0), 1);

                            // Distance histogram for matches
                            string hist_filename = baseDir + "/hist.jpg";
                            plotMatchDistanceHistogram(matching_result.match_distances, hist_filename);
                        }
                    } else {
                        result.stitching_success = false;
                        result.stitching_time_ms = 0;
                        result.output_size = cv::Size(0, 0);
                    }

                    auto end_total = chrono::high_resolution_clock::now();
                    auto duration = chrono::duration_cast<chrono::microseconds>(end_total - start_total);
                    result.total_time_ms = duration.count() / 1000.0;

                    results_.push_back(result);

                    cout << "  " << FeatureDetector::detectorTypeToString(detector_type)
                             << " + " << FeatureMatcher::matcherTypeToString(matcher_type)
                             << " + " << PanoramaStitcher::blendingTypeToString(blending_type)
                             << " (threshold=" << threshold << "): "
                             << result.keypoints_img1 + result.keypoints_img2 << " keypoints, "
                             << result.good_matches << " matches, "
                             << result.inliers_count << " inliers, "
                             << formatDuration(result.total_time_ms) << endl;
                }
            }
        }
    }

    cout << "Experiments completed. Total results: " << results_.size() << endl;
}

void Evaluator::runBatchExperiments(const vector<pair<cv::Mat, cv::Mat>>& image_pairs,
                                   const vector<string>& experiment_names) {
    for (size_t i = 0; i < image_pairs.size(); ++i) {
        string name = (i < experiment_names.size()) ? experiment_names[i] : "experiment_" + to_string(i);
        runExperiments(image_pairs[i].first, image_pairs[i].second, name);
    }
}

void Evaluator::saveResults(const string& filename) const {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << " for writing" << endl;
        return;
    }

    file << "detector_type,matcher_type,blending_type,reprojection_threshold,";
    file << "keypoints_img1,keypoints_img2,detection_time_ms,";
    file << "total_matches,good_matches,match_ratio,matching_time_ms,";
    file << "inliers_count,homography_time_ms,";
    file << "stitching_time_ms,output_width,output_height,stitching_success,";
    file << "total_time_ms" << endl;

    for (const auto& result : results_) {
        file << FeatureDetector::detectorTypeToString(result.detector_type) << ","
             << FeatureMatcher::matcherTypeToString(result.matcher_type) << ","
             << PanoramaStitcher::blendingTypeToString(result.blending_type) << ","
             << result.reprojection_threshold << ","
             << result.keypoints_img1 << ","
             << result.keypoints_img2 << ","
             << result.detection_time_ms << ","
             << result.total_matches << ","
             << result.good_matches << ","
             << result.match_ratio << ","
             << result.matching_time_ms << ","
             << result.inliers_count << ","
             << result.homography_time_ms << ","
             << result.stitching_time_ms << ","
             << result.output_size.width << ","
             << result.output_size.height << ","
             << (result.stitching_success ? "true" : "false") << ","
             << result.total_time_ms << endl;
    }

    file.close();
    cout << "Results saved to " << filename << endl;
}

void Evaluator::saveResultsAsTable(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return;
    }

    file << std::left << std::setw(8) << "Detector"
         << std::setw(10) << "Matcher"
         << std::setw(10) << "Blending"
         << std::setw(12) << "Threshold"
         << std::setw(12) << "Keypoints"
         << std::setw(10) << "Matches"
         << std::setw(10) << "Inliers"
         << std::setw(15) << "Total Time (ms)"
         << std::setw(10) << "Success" << std::endl;

    file << std::string(100, '-') << std::endl;

    for (const auto& result : results_) {
        file << std::left << std::setw(8) << ::FeatureDetector::detectorTypeToString(result.detector_type)
             << std::setw(10) << ::FeatureMatcher::matcherTypeToString(result.matcher_type)
             << std::setw(10) << ::PanoramaStitcher::blendingTypeToString(result.blending_type)
             << std::setw(12) << std::fixed << std::setprecision(1) << result.reprojection_threshold
             << std::setw(12) << (result.keypoints_img1 + result.keypoints_img2)
             << std::setw(10) << result.good_matches
             << std::setw(10) << result.inliers_count
             << std::setw(15) << std::fixed << std::setprecision(2) << result.total_time_ms
             << std::setw(10) << (result.stitching_success ? "Yes" : "No") << std::endl;
    }

    file.close();
    std::cout << "Table saved to " << filename << std::endl;
}

void Evaluator::generateReport(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
        return;
    }

    file << "# Panorama Stitching Experimental Results\n\n";
    file << "## Summary\n\n";
    file << "Total experiments conducted: " << results_.size() << "\n\n";

    auto successful_results = std::count_if(results_.begin(), results_.end(),
        [](const ExperimentResult& r) { return r.stitching_success; });

    file << "Successful stitching: " << successful_results << " / " << results_.size()
         << " (" << std::fixed << std::setprecision(1)
         << (100.0 * successful_results / results_.size()) << "%)\n\n";

    file << "## Performance Analysis\n\n";

    if (!results_.empty()) {
        auto best_result = *std::min_element(results_.begin(), results_.end(),
            [](const ExperimentResult& a, const ExperimentResult& b) {
                if (a.stitching_success != b.stitching_success) return a.stitching_success > b.stitching_success;
                return a.total_time_ms < b.total_time_ms;
            });

        file << "Best configuration:\n";
        file << "- Detector: " << ::FeatureDetector::detectorTypeToString(best_result.detector_type) << "\n";
        file << "- Matcher: " << ::FeatureMatcher::matcherTypeToString(best_result.matcher_type) << "\n";
        file << "- Blending: " << ::PanoramaStitcher::blendingTypeToString(best_result.blending_type) << "\n";
        file << "- Threshold: " << best_result.reprojection_threshold << "\n";
        file << "- Total Time: " << formatDuration(best_result.total_time_ms) << "\n";
        file << "- Inliers: " << best_result.inliers_count << "\n\n";
    }

    file.close();
    std::cout << "Report generated: " << filename << std::endl;
}

void Evaluator::saveVisualization(const cv::Mat& img1, const cv::Mat& img2,
                                 const std::vector<cv::KeyPoint>& keypoints1,
                                 const std::vector<cv::KeyPoint>& keypoints2,
                                 const std::vector<cv::DMatch>& matches,
                                  const std::string& filename) const {
    cv::Mat match_img;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, match_img);
    cv::imwrite(filename, match_img);
}

// Render a simple histogram image for a vector of distances.
static cv::Mat renderHistogram(const std::vector<double>& values, int bins = 50, cv::Size size = cv::Size(640, 360)) {
    if (values.empty()) return cv::Mat(size, CV_8UC3, cv::Scalar(255,255,255));
    double minv = *std::min_element(values.begin(), values.end());
    double maxv = *std::max_element(values.begin(), values.end());
    if (maxv <= minv) maxv = minv + 1.0;

    std::vector<int> hist(bins, 0);
    for (double v : values) {
        int b = static_cast<int>(bins * (v - minv) / (maxv - minv));
        if (b >= bins) b = bins - 1; if (b < 0) b = 0;
        hist[b]++;
    }

    int maxc = *std::max_element(hist.begin(), hist.end());
    cv::Mat canvas(size, CV_8UC3, cv::Scalar(255,255,255));
    int margin = 30;
    cv::Rect plot(margin, margin, size.width - 2*margin, size.height - 2*margin);
    cv::rectangle(canvas, plot, cv::Scalar(0,0,0));

    for (int i = 0; i < bins; ++i) {
        double ratio = maxc > 0 ? (double)hist[i] / maxc : 0.0;
        int h = static_cast<int>(ratio * plot.height);
        int x0 = plot.x + static_cast<int>((i / (double)bins) * plot.width);
        int x1 = plot.x + static_cast<int>(((i+1) / (double)bins) * plot.width) - 1;
        int y1 = plot.y + plot.height - 1;
        int y0 = y1 - h;
        cv::rectangle(canvas, cv::Point(x0,y0), cv::Point(x1,y1), cv::Scalar(80,140,240), cv::FILLED);
    }

    // axes labels (min/max)
    cv::putText(canvas, std::to_string(minv), cv::Point(plot.x, plot.y + plot.height + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 1, cv::LINE_AA);
    cv::putText(canvas, std::to_string(maxv), cv::Point(plot.x + plot.width - 60, plot.y + plot.height + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0,0,0), 1, cv::LINE_AA);
    return canvas;
}

void Evaluator::plotMatchDistanceHistogram(const std::vector<double>& distances,
                                           const std::string& filename,
                                           const std::string&) const {
    cv::Mat hist = renderHistogram(distances, 50, cv::Size(640, 360));
    if (!hist.empty()) cv::imwrite(filename, hist);
}

void Evaluator::saveKeypointsCross(const cv::Mat& img,
                                   const std::vector<cv::KeyPoint>& keypoints,
                                   const std::string& filename,
                                   const cv::Scalar& color,
                                   int thickness) const {
    if (img.empty()) return;
    cv::Mat canvas;
    if (img.channels() == 1) {
        cv::cvtColor(img, canvas, cv::COLOR_GRAY2BGR);
    } else {
        canvas = img.clone();
    }

    for (const auto& kp : keypoints) {
        const cv::Point2f c = kp.pt;
        int half = std::max(3, static_cast<int>(std::round(kp.size * 0.5)));
        cv::Point p1(cvRound(c.x - half), cvRound(c.y - half));
        cv::Point p2(cvRound(c.x + half), cvRound(c.y + half));
        cv::Point p3(cvRound(c.x - half), cvRound(c.y + half));
        cv::Point p4(cvRound(c.x + half), cvRound(c.y - half));
        cv::line(canvas, p1, p2, color, thickness, cv::LINE_AA);
        cv::line(canvas, p3, p4, color, thickness, cv::LINE_AA);
    }

    cv::imwrite(filename, canvas);
}

std::string Evaluator::formatDuration(double ms) const {
    if (ms < 1000) {
        return std::to_string(static_cast<int>(ms)) + "ms";
    } else {
        return std::to_string(ms / 1000.0) + "s";
    }
}

void Evaluator::ensureOutputDirectory() const {
    if (!config_.output_dir.empty()) {
        std::filesystem::create_directories(config_.output_dir);
    }
}

void Evaluator::saveIntermediateImage(const cv::Mat& image, const std::string& filename) const {
    if (!image.empty()) {
        cv::imwrite(filename, image);
    }
}
