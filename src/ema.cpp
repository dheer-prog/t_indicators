#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>
#include<iostream>
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



void compute_ema_1d(const float* data,
                    py::ssize_t len,
                    py::ssize_t in_stride,
                    size_t window,
                    float* out,
                    py::ssize_t out_stride) {
    

    if (window == 0 || static_cast<py::ssize_t>(window) > len) {
        return;
    }

    const py::ssize_t signed_window = static_cast<py::ssize_t>(window);
    const float alpha = 2.0f / (static_cast<float>(window) + 1.0f);
    const float beta = 1.0f - alpha;
    float sum = 0.0f;
    for (py::ssize_t i = 0; i < signed_window; ++i) {
        sum += data[i * in_stride];
    }
    out[(signed_window - 1) * out_stride] = sum / static_cast<float>(window);

    for (py::ssize_t i = signed_window; i < len; ++i) {
        out[i * out_stride] = alpha * data[i * in_stride] + beta * out[(i - 1) * out_stride];
    }
}

void compute_ema_matrix_by_column(const MatrixView& input, size_t window, float* out) {
    const py::ssize_t rows = input.rows;
    const py::ssize_t cols = input.cols;
    const size_t col_count = static_cast<size_t>(cols);

    std::fill_n(out, rows * cols, nan_value());
    if (window == 0 || static_cast<py::ssize_t>(window) > rows) {
        return;
    }

    const unsigned int hw = std::max(1u, std::thread::hardware_concurrency());
    std::cout<<hw;
    const size_t num_threads = (col_count >= 32) ? std::min<size_t>(hw, col_count) : 1;
    const size_t chunk = (col_count + num_threads - 1) / num_threads;

    auto worker = [&](size_t start_col, size_t end_col) {
        for (size_t c = start_col; c < end_col; ++c) {
            compute_ema_1d(input.data + static_cast<py::ssize_t>(c) * input.col_stride,
                           rows,
                           input.row_stride,
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

py::object ema_series(py::object series, int window) {
    py::array input = series.attr("to_numpy")(py::arg("dtype") = py::dtype::of<float>(),
                                              py::arg("copy") = false);
    auto info = input.request();

    if (info.ndim != 1) {
        throw std::runtime_error("Expected a 1D pandas Series.");
    }

    py::array_t<float> output(info.shape[0]);
    auto out_info = output.request();

    {
        py::gil_scoped_release release_gil;
        std::fill_n(static_cast<float*>(out_info.ptr),out_info.shape[0],nan_value());
        compute_ema_1d(static_cast<const float*>(info.ptr),
                       info.shape[0],
                       info.strides[0] / static_cast<py::ssize_t>(sizeof(float)),
                       static_cast<size_t>(window),
                       static_cast<float*>(out_info.ptr),
                       1);
    }

    return pandas_module().attr("Series")(output,
                                          py::arg("index") = series.attr("index"),
                                          py::arg("name") = series.attr("name"));
}

py::object ema_dataframe(py::object data_frame, int window) {
    py::array input = data_frame.attr("to_numpy")(py::arg("dtype") = py::dtype::of<float>(),
                                                  py::arg("copy") = false);
    auto info = input.request();

    if (info.ndim != 2) {
        throw std::runtime_error("Expected a 2D pandas DataFrame.");
    }

    const py::ssize_t rows = info.shape[0];
    const py::ssize_t cols = info.shape[1];
    MatrixView view{
        static_cast<const float*>(info.ptr),
        rows,
        cols,
        info.strides[0] / static_cast<py::ssize_t>(sizeof(float)),
        info.strides[1] / static_cast<py::ssize_t>(sizeof(float)),
    };

    py::array_t<float> output({rows, cols});
    auto out_info = output.request();

    {
        py::gil_scoped_release release_gil;
        compute_ema_matrix_by_column(view, static_cast<size_t>(window), static_cast<float*>(out_info.ptr));
    }

    return pandas_module().attr("DataFrame")(output,
                                             py::arg("index") = data_frame.attr("index"),
                                             py::arg("columns") = data_frame.attr("columns"));
}

void register_ema(py::module_& m) {
    m.def("ema_series", &ema_series, py::arg("series"), py::arg("window"));
    m.def("ema", &ema_dataframe, py::arg("dataframe"), py::arg("window"));
}
