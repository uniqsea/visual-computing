#pragma once

#include "filters/Filter.h"
#include "filters/PixelationFilter.h"
#include "filters/ComicArtFilter.h"
#include "filters/OilPaintingFilter.h"
#include <memory>

enum class ProcessingMode {
    CPU,
    GPU
};

class FilterManager {
public:
    FilterManager();
    
    void setCurrentFilter(FilterType type);
    void setProcessingMode(ProcessingMode mode);
    
    FilterType getCurrentFilterType() const { return currentFilterType; }
    ProcessingMode getProcessingMode() const { return processingMode; }
    
    Filter* getCurrentFilter() { return currentFilter; }
    
    std::string getCurrentFilterName() const;
    std::string getProcessingModeName() const;
    
private:
    FilterType currentFilterType;
    ProcessingMode processingMode;
    
    Filter* currentFilter;
    std::unique_ptr<PixelationFilter> pixelationFilter;
    std::unique_ptr<CartoonFilter> cartoonFilter;
    std::unique_ptr<OilPaintingFilter> oilPaintingFilter;
    
    void updateCurrentFilter();
};
