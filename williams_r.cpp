#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <limits>
using namespace std;
namespace py = pybind11;

py::array_t<float> williams_r(const py::array_t<float>& data, int window)
{
    py::buffer_info buf = data.request();
    if (buf.ndim != 1) throw std::runtime_error("Data must be 1D");
    float* ptr = static_cast<float*>(buf.ptr);
    size_t size = buf.shape[0];

    // Create result array
    py::array_t<float> result = py::array_t<float>(size);
    py::buffer_info res_buf = result.request(true); // writable
    float* res_ptr = static_cast<float*>(res_buf.ptr);

    // Initialize to NAN
    std::fill(res_ptr, res_ptr + size, std::numeric_limits<float>::quiet_NaN());

    if (window <= 0 || size < window) return result;

    for (size_t i = window - 1; i < size; ++i) {
        float highest_high = -std::numeric_limits<float>::infinity();
        float lowest_low = std::numeric_limits<float>::infinity();
        for (size_t j = i - window + 1; j <= i; ++j) {
            if (ptr[j] > highest_high) highest_high = ptr[j];
            if (ptr[j] < lowest_low) lowest_low = ptr[j];
        }
        float denom = highest_high - lowest_low;
        if (denom == 0) {
            res_ptr[i] = std::numeric_limits<float>::quiet_NaN();
        } else {
            res_ptr[i] = (highest_high - ptr[i]) / denom * -100.0f;
        }
    }
    return result;
}

py::object williams_r_series(py::object series, int window)
{
    py::module_ pd = py::module_::import("pandas");
    py::array_t<float> values = series.attr("values").cast<py::array_t<float>>();
    py::object index = series.attr("index");
    py::array_t<float> williams = williams_r(values, window);
    return pd.attr("Series")(williams, py::arg("index") = index);
}

py::object williams_r_dataframe(py::object dataframe, int window)
{
    py::module_ pd = py::module_::import("pandas");
    py::dict new_df;
    py::object cols = dataframe.attr("columns");
    for (py::handle col : cols) {
        py::object series_col = dataframe.attr("__getitem__")(col);
        py::object wr_series_result = williams_r_series(series_col, window);
        new_df[col] = wr_series_result;
    }
    return pd.attr("DataFrame")(new_df, py::arg("index") = dataframe.attr("index"));
}

void register_williams_r(py::module_ &m)
{
    m.def("williams_r", &williams_r_series, "Computing Williams %R of series");
    m.def("williams_r_DataFrame", &williams_r_dataframe, "Calculate Williams %R of entire dataframe");
}










