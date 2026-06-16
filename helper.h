#pragma once

#include <limits>

inline float nan_value() {
    return std::numeric_limits<float>::quiet_NaN();
}
