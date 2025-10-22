#include "FilterManager.h"

FilterManager::FilterManager() 
    : currentFilterType(FilterType::None), 
      processingMode(ProcessingMode::GPU) {
    
    // Initialize filters
    pixelationFilter = std::make_unique<PixelationFilter>();
    cartoonFilter = std::make_unique<CartoonFilter>();
    
    updateCurrentFilter();
}

void FilterManager::setCurrentFilter(FilterType type) {
    currentFilterType = type;
    updateCurrentFilter();
}

void FilterManager::setProcessingMode(ProcessingMode mode) {
    processingMode = mode;
}

void FilterManager::updateCurrentFilter() {
    switch (currentFilterType) {
        case FilterType::Pixelation:
            currentFilter = std::unique_ptr<Filter>(pixelationFilter.get());
            break;
        case FilterType::Cartoon:
            currentFilter = std::unique_ptr<Filter>(cartoonFilter.get());
            break;
        case FilterType::None:
        default:
            currentFilter.reset();
            break;
    }
}

std::string FilterManager::getCurrentFilterName() const {
    switch (currentFilterType) {
        case FilterType::None:
            return "None";
        case FilterType::Pixelation:
            return "Pixelation";
        case FilterType::Cartoon:
            return "Cartoon";
        default:
            return "Unknown";
    }
}

std::string FilterManager::getProcessingModeName() const {
    return (processingMode == ProcessingMode::CPU) ? "CPU" : "GPU";
}

