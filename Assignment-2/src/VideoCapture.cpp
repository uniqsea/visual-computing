#include "VideoCapture.h"
#include <iostream>

VideoCapture::VideoCapture() : currentWidth(0), currentHeight(0) {
}

VideoCapture::~VideoCapture() {
    close();
}

bool VideoCapture::open(int deviceID) {
    capture.open(deviceID);
    if (!capture.isOpened()) {
        std::cerr << "Failed to open camera with device ID: " << deviceID << std::endl;
        return false;
    }
    
    // Get current resolution
    currentWidth = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    currentHeight = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    std::cout << "Camera opened successfully. Default resolution: " 
              << currentWidth << "x" << currentHeight << std::endl;
    
    return true;
}

void VideoCapture::close() {
    if (capture.isOpened()) {
        capture.release();
    }
}

bool VideoCapture::isOpened() const {
    return capture.isOpened();
}

bool VideoCapture::getFrame(cv::Mat& frame) {
    if (!capture.isOpened()) {
        return false;
    }
    
    capture >> frame;
    return !frame.empty();
}

bool VideoCapture::setResolution(int width, int height) {
    if (!capture.isOpened()) {
        std::cerr << "Cannot set resolution: camera not opened" << std::endl;
        return false;
    }
    
    capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    
    // Verify the actual resolution set (camera may not support exact resolution)
    currentWidth = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    currentHeight = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    std::cout << "Resolution set to: " << currentWidth << "x" << currentHeight << std::endl;
    
    return (currentWidth == width && currentHeight == height);
}

void VideoCapture::getResolution(int& width, int& height) const {
    width = currentWidth;
    height = currentHeight;
}

bool VideoCapture::setResolution640x480() {
    return setResolution(640, 480);
}

bool VideoCapture::setResolution1280x720() {
    return setResolution(1280, 720);
}

bool VideoCapture::setResolution1920x1080() {
    return setResolution(1920, 1080);
}

std::string VideoCapture::getResolutionString() const {
    return std::to_string(currentWidth) + "x" + std::to_string(currentHeight);
}

