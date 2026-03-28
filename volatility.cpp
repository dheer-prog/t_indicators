#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
using namespace std;
namespace py=pybind11;
#include"probability.h"

vector<float> returns(const vector<float>& data)
{
    vector<float> ret(data.size(),NAN);
    for(int i=1;i<data.size();i++)
    {
        ret[i]=(data[i]-data[i-1])/data[i-1];
    }
    return ret;
}
vector<float> fast_volatility(const vector<float>& data, int window)
{
    int n = data.size();
    vector<float> vol(n, NAN);

    if (n < window + 1) return vol;

    // Step 1: compute returns
    vector<float> ret(n, NAN);
    for (int i = 1; i < n; i++) {
        if (data[i-1] != 0)
            ret[i] = (data[i] - data[i-1]) / data[i-1];
    }

    float sum = 0.0f, sum_sq = 0.0f;

    // Step 2: initialize first window
    for (int i = 1; i <= window; i++) {
        sum += ret[i];
        sum_sq += ret[i] * ret[i];
    }

    // Step 3: rolling update
    for (int i = window; i < n; i++) {
        float mean = sum / window;
        float variance = (sum_sq - (sum * sum) / window) / (window - 1);

        vol[i] = sqrt(variance);

        // slide window
        if (i + 1 < n) {
            float new_val = ret[i + 1];
            float old_val = ret[i - window + 1];

            sum += new_val - old_val;
            sum_sq += new_val * new_val - old_val * old_val;
        }
    }

    return vol;
}
py::object volatility_series(py::object series, int window) {
    py::module_ np = py::module_::import("numpy");
    py::module_ pd = py::module_::import("pandas");

    auto values = np.attr("asarray")(series).cast<std::vector<float>>();
    auto index = series.attr("index");

    auto result = fast_volatility(values, window);
    py::array result_array = py::cast(result);
    
    return pd.attr("Series")(result_array, py::arg("index") = index);
}
py::object volatility_dataframe(py::object dataframe, int window) {
    py::module_ np = py::module_::import("numpy");
    py::module_ pd = py::module_::import("pandas");

    auto index = dataframe.attr("index");
    auto columns = dataframe.attr("columns");
    py::dict new_df; 

    for (auto& column : columns) {
        auto series = dataframe.attr("__getitem__")(column);
        auto values = np.attr("asarray")(series).cast<std::vector<float>>();
        auto vol_series = fast_volatility(values, window);
        py::object vol_array = py::cast(vol_series);
        new_df[column] = vol_array;
    }
    
    return pd.attr("DataFrame")(new_df,py::arg("index") = index);
}
void register_volatility(py::module_ &m)
{
    m.def("volatility_series", &volatility_series, py::arg("series"), py::arg("window"));
    m.def("volatility_dataframe", &volatility_dataframe, py::arg("dataframe"), py::arg("window"));
    m.def("returns", &returns, py::arg("series"));
}
