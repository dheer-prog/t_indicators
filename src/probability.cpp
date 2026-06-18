//This contains the most commonly used functions in probability theory

#include<vector>
#include<cmath>
#include<numeric>
#include<limits>  
#include<algorithm>
#include<iostream>
#include "include/probability.h"
float probabilities::std_dev(const std::vector<float>& data) {
    int n = data.size();
    if (n <= 1) return 0.0;
    float mean = std::accumulate(data.begin(), data.end(), 0.0) / n;
    float sum_sq = 0.0;
    for (float x : data)
        sum_sq += (x - mean) * (x - mean);
    return std::sqrt(sum_sq / (n - 1));
}
