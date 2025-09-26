#pragma once

#include <opencv2/opencv.hpp>
#include <string>

enum class WarperType {
    NONE,
    CYLINDRICAL,
    SPHERICAL
};

class ImageWarper {
private:
    WarperType type_;
    double focal_;

public:
    ImageWarper(WarperType type = WarperType::NONE, double focal = 0.0)
        : type_(type), focal_(focal) {}

    void setType(WarperType t) { type_ = t; }
    void setFocal(double f) { focal_ = f; }
    WarperType getType() const { return type_; }
    double getFocal() const { return focal_; }

    cv::Mat warp(const cv::Mat& image) const;

    static std::string warperTypeToString(WarperType t);

private:
    cv::Mat warpCylindrical(const cv::Mat& image) const;
    cv::Mat warpSpherical(const cv::Mat& image) const;
};

