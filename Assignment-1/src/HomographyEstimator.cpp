#include "HomographyEstimator.hpp"
#include <iostream>
#include <chrono>

using namespace std;

HomographyEstimator::HomographyEstimator(double reprojection_threshold, double confidence, int max_iterations)
    : reprojection_threshold_(reprojection_threshold), confidence_(confidence), max_iterations_(max_iterations) {
}

HomographyResult HomographyEstimator::estimateHomography(
    const vector<cv::Point2f>& points1,
    const vector<cv::Point2f>& points2) {

    HomographyResult result;
    result.reprojection_threshold = reprojection_threshold_;
    result.confidence = confidence_;
    result.valid = false;

    if (points1.size() < 4 || points2.size() < 4 || points1.size() != points2.size()) {
        cerr << "HomographyEstimator: Need at least 4 point pairs for homography estimation" << endl;
        return result;
    }

    auto start_time = chrono::high_resolution_clock::now();

    try {
        result.homography = cv::findHomography(
            points1, points2,
            cv::RANSAC,
            reprojection_threshold_,
            result.inliers_mask,
            max_iterations_,
            confidence_
        );

        if (!result.homography.empty()) {
            result.valid = true;
            result.inliers_count = cv::countNonZero(result.inliers_mask);
        }
    } catch (const cv::Exception& e) {
        cerr << "HomographyEstimator: OpenCV error - " << e.what() << endl;
    }

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
    result.estimation_time_ms = duration.count() / 1000.0;

    return result;
}

HomographyResult HomographyEstimator::estimateHomographyFromMatches(
    const vector<cv::KeyPoint>& keypoints1,
    const vector<cv::KeyPoint>& keypoints2,
    const vector<cv::DMatch>& matches) {

    vector<cv::Point2f> points1 = extractPoints(keypoints1, matches, true);
    vector<cv::Point2f> points2 = extractPoints(keypoints2, matches, false);

    return estimateHomography(points1, points2);
}

vector<cv::Point2f> HomographyEstimator::extractPoints(
    const vector<cv::KeyPoint>& keypoints,
    const vector<cv::DMatch>& matches,
    bool query_points) {

    vector<cv::Point2f> points;
    points.reserve(matches.size());

    for (const auto& match : matches) {
        if (query_points) {
            if (match.queryIdx < static_cast<int>(keypoints.size())) {
                points.push_back(keypoints[match.queryIdx].pt);
            }
        } else {
            if (match.trainIdx < static_cast<int>(keypoints.size())) {
                points.push_back(keypoints[match.trainIdx].pt);
            }
        }
    }

    return points;
}
