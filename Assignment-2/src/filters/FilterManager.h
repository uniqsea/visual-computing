#pragma once

#include "Filter.h"
#include "PixelationFilter.h"
#include "CartoonFilter.h"
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
    
    Filter* getCurrentFilter() { return currentFilter.get(); }
    
    std::string getCurrentFilterName() const;
    std::string getProcessingModeName() const;
    
private:
    FilterType currentFilterType;
    ProcessingMode processingMode;
    
    std::unique_ptr<Filter> currentFilter;
    std::unique_ptr<PixelationFilter> pixelationFilter;
    std::unique_ptr<CartoonFilter> cartoonFilter;
    
    void updateCurrentFilter();
};

