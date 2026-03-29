#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;

py::array_t<float> rolling_sma(const py::array_t<float>& data, int window)
{
    py::buffer_info buf = data.request();
    if (buf.ndim != 1) throw std::runtime_error("Data must be 1D");
    float* ptr = static_cast<float*>(buf.ptr);
    size_t size = buf.shape[0];
    float sum = 0.0;

    py::array_t<float> result = py::array_t<float>(size);
    // Create result numpy array
    py::buffer_info res_buf = result.request(true); // writable
    //Gives me a pointer to results array with all its info such as ndim,shape,and a raw ptr
    float* res_ptr = static_cast<float*>(res_buf.ptr);
    //Gives me a raw float ptr
    std::fill(res_ptr, res_ptr + size, std::numeric_limits<float>::quiet_NaN());
    //fills results with NAN
     if(window<=0)
    {
        return result;
    }
    if (window == 1) {
        for (size_t i = 0; i < size; i++) {
            res_ptr[i] = ptr[i];
        }
        return result;
    }
    if (window > size) {
        return result;
    }
    for (size_t i = 0; i < size; i++) {
        if (i < window) {
            sum = sum + ptr[i];
            continue;
        }
        if (i >= window) {
            sum = sum - ptr[i - window];
        }
        if (i >= window - 1) {
            res_ptr[i] = sum / window;
        }
    }
    return result;
}
py::object rolling_sma_series(py::object series, int window)
{
    py::module_ pd = py::module_::import("pandas");
    py::array_t<float> values = series.attr("values").cast<py::array_t<float>>();
    py::object index = series.attr("index");
    py::array_t<float> result = rolling_sma(values, window);
    return pd.attr("Series")(result, py::arg("index") = index);
}
py::object rolling_sma_dataframe(py::object data_frame, int window)
{
    py::module_ pd = py::module_::import("pandas");
    py::dict new_df;
    py::object cols = data_frame.attr("columns");
    for (py::handle col : cols) {
        py::object series_col = data_frame.attr("__getitem__")(col);
        py::object sma_series = rolling_sma_series(series_col, window);
        new_df[col] = sma_series;
    }
    return pd.attr("DataFrame")(new_df, py::arg("index") = data_frame.attr("index"));
}
void register_rolling(py::module_ &m)
{
    m.def("rolling_sma_series", &rolling_sma_series, "Compute rolling SMA of a series");
    m.def("rolling_sma_dataframe", &rolling_sma_dataframe, "Compute rolling SMA of the entire dataframe");
}