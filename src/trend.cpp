/*
Trend is the difference between EWMA's
So trend_5_20 is the difference between
EWMA 20 and EWMA 5 

Hence this also contains EWMA function*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
using namespace std;
namespace py=pybind11;

float compute_std(const std::vector<float>& data) {
    int n = data.size();
    if (n <= 1) return 0.0;
    float mean = std::accumulate(data.begin(), data.end(), 0.0) / n;
    float sum_sq = 0.0;
    for (float x : data)
        sum_sq += (x - mean) * (x - mean);
    return std::sqrt(sum_sq / (n - 1));
}

vector<float> EWMA(const vector<float> data,int window)
{
    vector<float> EWMA(data.size(),NAN);
    float sum=0;
    for(int i=0;i<window;i++)
    {
        sum=sum+data[i];
    }
    EWMA[window-1]=sum/window;
    float alpha=2/(window+1);
    for(int i=window;i<data.size();i++)
    {
        EWMA[i]=(alpha*data[i])+((1-alpha)*EWMA[i-1]);
    }
    return EWMA;
}

vector<float> trend(const vector<float>& data, int small_window,int large_window)
{
    vector<float> trend;
    vector<float> EWMA_small=EWMA(data,small_window);
    vector<float> EWMA_large=EWMA(data,large_window);
    float std=compute_std(EWMA_large);
    for(int i=0;i<data.size();i++)
    {
        trend.push_back((EWMA_small[i]-EWMA_large[i])/std);
    }
    return trend; 

}
 
py::object ewma_series(py::object series, int window) {
    py::module_ np = py::module_::import("numpy");
    py::module_ pd = py::module_::import("pandas");

    auto values = np.attr("asarray")(series).cast<std::vector<float>>();
    auto index = series.attr("index");

    auto result = EWMA(values, window);
    py::array result_array = py::cast(result);
    return pd.attr("Series")(result_array, py::arg("index") = index);
}

py::object ewma_dataframe(py::object dataframe, int window) {
    py::module_ pd = py::module_::import("pandas");
    py::dict new_df;

    auto cols = dataframe.attr("columns");
    for (auto col : cols) {
        py::object series_col = dataframe.attr("__getitem__")(col);
        py::object ewma_result = ewma_series(series_col, window);
        new_df[col] = ewma_result;
    }

    return pd.attr("DataFrame")(new_df, py::arg("index") = dataframe.attr("index"));
}

py::object trend_series(py::object series, int small_window, int large_window) {
    py::module_ np = py::module_::import("numpy");
    py::module_ pd = py::module_::import("pandas");

    auto values = np.attr("asarray")(series).cast<std::vector<float>>();
    auto index = series.attr("index");

    auto trend_result = trend(values, small_window, large_window);
    py::array result_array = py::cast(trend_result);
    return pd.attr("Series")(result_array, py::arg("index") = index);
}

py::object trend_dataframe(py::object dataframe, int small_window, int large_window) {
    py::module_ pd = py::module_::import("pandas");
    py::dict new_df;

    auto cols = dataframe.attr("columns");
    for (auto col : cols) {
        py::object series_col = dataframe.attr("__getitem__")(col);
        py::object trend_result = trend_series(series_col, small_window, large_window);
        new_df[col] = trend_result;
    }

    return pd.attr("DataFrame")(new_df, py::arg("index") = dataframe.attr("index"));
}

void register_trend_ewma(py::module_ &m)
{
    m.def("ewma", &ewma_series,"Computing RSI of series ");
    m.def("EWMA_DataFrame",&ewma_dataframe,"Calculate RSI of entire dataframe");
    m.def("trend",&trend_series,"Calculating trend");
    m.def("trend_DataFrame",&trend_dataframe,"Calculating trend in dataframe");

}

        

   

