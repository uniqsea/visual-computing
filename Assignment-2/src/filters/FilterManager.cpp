#include "filters/FilterManager.h"

FilterManager::FilterManager() 
    : currentFilterType(FilterType::None), 
      processingMode(ProcessingMode::GPU),
      currentFilter(nullptr) {
    
    // Initialize filters
    pixelationFilter = std::make_unique<PixelationFilter>();
    cartoonFilter = std::make_unique<CartoonFilter>();
    oilPaintingFilter = std::make_unique<OilPaintingFilter>();
    
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
            currentFilter = pixelationFilter.get();
            break;
        case FilterType::Cartoon:
            currentFilter = cartoonFilter.get();
            break;
        case FilterType::OilPainting:
            currentFilter = oilPaintingFilter.get();
            break;
        case FilterType::None:
        default:
            currentFilter = nullptr;
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
            return "Comic Art";
        case FilterType::OilPainting:
            return "OilPainting";
        default:
            return "Unknown";
    }
}

std::string FilterManager::getProcessingModeName() const {
    return (processingMode == ProcessingMode::CPU) ? "CPU" : "GPU";
}
