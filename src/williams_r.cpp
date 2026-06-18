#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>
#include "include/helper.h"

namespace py = pybind11;

namespace {

struct MatrixView {
    const float* data;
    py::ssize_t rows;
    py::ssize_t cols;
    py::ssize_t row_stride;
    py::ssize_t col_stride;
};

void compute_williams_1d(
                    const float* data_high,
                    const float* data_low,
                    const float* data_close,
                    py::ssize_t len,
                    py::ssize_t in_stride,
                    py::ssize_t window,
                    float* out,
                    py::ssize_t out_stride
) {
    if (window == 0 || static_cast<py::ssize_t>(window) > len) {
        return;
    }

    std::multiset<float> vals;
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(window); i++) {
        vals.insert(data_high[i * in_stride]);
        vals.insert(data_low[i * in_stride]);
    }
    float highest = *vals.rbegin();
    float lowest = *vals.begin();
    float denom = 0;
    if (highest == lowest) {
        out[(static_cast<py::ssize_t>(window) - 1) * out_stride] = nan_value();
    } else {
        denom = 1 / (highest - lowest);
        out[(static_cast<py::ssize_t>(window) - 1) * out_stride] =
            (highest - data_close[(static_cast<py::ssize_t>(window) - 1) * in_stride]) * denom * (-100.0f);
    }

    for (py::ssize_t i = static_cast<py::ssize_t>(window); i < len; i++) {
        vals.insert(data_high[i*in_stride]); 
        vals.insert(data_low[i*in_stride]); 
        vals.erase(vals.find(data_high[(i - static_cast<py::ssize_t>(window)) * in_stride]));
        vals.erase(vals.find(data_low[(i - static_cast<py::ssize_t>(window)) * in_stride]));
        float nhigh = *vals.rbegin();
        float nlow = *vals.begin();
        if (nhigh != highest || nlow != lowest) {
            highest = nhigh;
            lowest = nlow;
            if (highest == lowest) {
                out[i * out_stride] = nan_value();
                continue;
            }
            denom = 1 / (highest - lowest);
        }
        if (highest == lowest) {
            out[i * out_stride] = nan_value();
        } else {
            out[i * out_stride] = (highest - data_close[i * in_stride]) * denom * (-100.0f);
        }
    }
}

void compute_williams_matrix_by_column(const MatrixView& input_high,
                                       const MatrixView& input_low,
                                       const MatrixView& input_close,
                                       py::ssize_t window,
                                       float* out) {
    const py::ssize_t rows = input_high.rows;
    const py::ssize_t cols = input_high.cols;
    const size_t col_count = static_cast<size_t>(cols);

    std::fill_n(out, rows * cols, nan_value());
    if (window == 0 || window > rows) {
        return;
    }

    const unsigned int hw = std::max(1u, std::thread::hardware_concurrency());
    const size_t num_threads = (col_count >= 32) ? std::min<size_t>(hw, col_count) : 1;
    const size_t chunk = (col_count + num_threads - 1) / num_threads;

    auto worker = [&](size_t start_col, size_t end_col) {
        for (size_t c = start_col; c < end_col; ++c) {
            compute_williams_1d(input_high.data + static_cast<py::ssize_t>(c) * input_high.col_stride,
                                input_low.data + static_cast<py::ssize_t>(c) * input_low.col_stride,
                                input_close.data + static_cast<py::ssize_t>(c) * input_close.col_stride,
                                rows,
                                input_high.row_stride,
                                window,
                                out + static_cast<py::ssize_t>(c),
                                cols);
        }
    };

    if (num_threads == 1) {
        worker(0, col_count);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    for (size_t t = 0; t < num_threads; ++t) {
        const size_t start = t * chunk;
        const size_t end = std::min(start + chunk, col_count);
        if (start >= end) {
            break;
        }
        workers.emplace_back(worker, start, end);
    }

    for (auto& thread : workers) {
        thread.join();
    }
}

py::module_ pandas_module() {
    return py::module_::import("pandas");
}

}  // namespace

py::object williams_r_series(py::object data_high, py::object data_low, py::object data_close, int window) {
    py::array high_input = data_high.attr("to_numpy")(py::arg("dtype") = py::dtype::of<float>(),
                                                      py::arg("copy") = false);
    py::array low_input = data_low.attr("to_numpy")(py::arg("dtype") = py::dtype::of<float>(),
                                                    py::arg("copy") = false);
    py::array close_input = data_close.attr("to_numpy")(py::arg("dtype") = py::dtype::of<float>(),
                                                        py::arg("copy") = false);
    auto high_info = high_input.request();
    auto low_info = low_input.request();
    auto close_info = close_input.request();

    if (high_info.ndim != 1 || low_info.ndim != 1 || close_info.ndim != 1) {
        throw std::runtime_error("Expected 1D pandas Series for high, low, and close.");
    }
    if (high_info.shape[0] != low_info.shape[0] || high_info.shape[0] != close_info.shape[0]) {
        throw std::runtime_error("High, low, and close series must have the same length.");
    }

    const py::ssize_t high_stride =
        high_info.strides[0] / static_cast<py::ssize_t>(sizeof(float));
    const py::ssize_t low_stride =
        low_info.strides[0] / static_cast<py::ssize_t>(sizeof(float));
    const py::ssize_t close_stride =
        close_info.strides[0] / static_cast<py::ssize_t>(sizeof(float));
    if (high_stride != low_stride || high_stride != close_stride) {
        throw std::runtime_error("High, low, and close series must have matching strides.");
    }

    py::array_t<float> output(high_info.shape[0]);
    auto out_info = output.request();

    {
        py::gil_scoped_release release_gil;
        std::fill_n(static_cast<float*>(out_info.ptr), out_info.shape[0], nan_value());
        compute_williams_1d(static_cast<const float*>(high_info.ptr),
                            static_cast<const float*>(low_info.ptr),
                            static_cast<const float*>(close_info.ptr),
                            high_info.shape[0],
                            high_stride,
                            static_cast<py::ssize_t>(window),
                            static_cast<float*>(out_info.ptr),
                            1);
    }

    return pandas_module().attr("Series")(output,
                                          py::arg("index") = data_close.attr("index"),
                                          py::arg("name") = data_close.attr("name"));
}

py::object williams_r_dataframe(py::object data_high, py::object data_low, py::object data_close, int window) {
    py::array high_input = data_high.attr("to_numpy")(py::arg("dtype") = py::dtype::of<float>(),
                                                      py::arg("copy") = false);
    py::array low_input = data_low.attr("to_numpy")(py::arg("dtype") = py::dtype::of<float>(),
                                                    py::arg("copy") = false);
    py::array close_input = data_close.attr("to_numpy")(py::arg("dtype") = py::dtype::of<float>(),
                                                        py::arg("copy") = false);
    auto high_info = high_input.request();
    auto low_info = low_input.request();
    auto close_info = close_input.request();

    if (high_info.ndim != 2 || low_info.ndim != 2 || close_info.ndim != 2) {
        throw std::runtime_error("Expected 2D pandas DataFrame inputs for high, low, and close.");
    }
    if (high_info.shape[0] != low_info.shape[0] || high_info.shape[0] != close_info.shape[0] ||
        high_info.shape[1] != low_info.shape[1] || high_info.shape[1] != close_info.shape[1]) {
        throw std::runtime_error("High, low, and close dataframes must have the same shape.");
    }

    MatrixView high_view{
        static_cast<const float*>(high_info.ptr),
        high_info.shape[0],
        high_info.shape[1],
        high_info.strides[0] / static_cast<py::ssize_t>(sizeof(float)),
        high_info.strides[1] / static_cast<py::ssize_t>(sizeof(float)),
    };
    MatrixView low_view{
        static_cast<const float*>(low_info.ptr),
        low_info.shape[0],
        low_info.shape[1],
        low_info.strides[0] / static_cast<py::ssize_t>(sizeof(float)),
        low_info.strides[1] / static_cast<py::ssize_t>(sizeof(float)),
    };
    MatrixView close_view{
        static_cast<const float*>(close_info.ptr),
        close_info.shape[0],
        close_info.shape[1],
        close_info.strides[0] / static_cast<py::ssize_t>(sizeof(float)),
        close_info.strides[1] / static_cast<py::ssize_t>(sizeof(float)),
    };

    if (high_view.row_stride != low_view.row_stride || high_view.row_stride != close_view.row_stride ||
        high_view.col_stride != low_view.col_stride || high_view.col_stride != close_view.col_stride) {
        throw std::runtime_error("High, low, and close dataframes must have matching strides.");
    }

    py::array_t<float> output({high_view.rows, high_view.cols});
    auto out_info = output.request();

    {
        py::gil_scoped_release release_gil;
        compute_williams_matrix_by_column(high_view,
                                          low_view,
                                          close_view,
                                          static_cast<py::ssize_t>(window),
                                          static_cast<float*>(out_info.ptr));
    }

    return pandas_module().attr("DataFrame")(output,
                                             py::arg("index") = data_close.attr("index"),
                                             py::arg("columns") = data_close.attr("columns"));
}

void register_williams_r(py::module_& m)
{
    m.def("williams_r", &williams_r_series, "Computing Williams %R of series");
    m.def("williams_r_DataFrame", &williams_r_dataframe, "Calculate Williams %R of entire dataframe");
}
