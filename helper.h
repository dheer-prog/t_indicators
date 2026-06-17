#pragma once

#include <limits>
inline float inf_val(){
    return std::numeric_limits<float>::infinity();
}
inline float nan_value() {
    return std::numeric_limits<float>::quiet_NaN();
}
