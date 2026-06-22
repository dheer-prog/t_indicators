#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <vector>
#include "include/ema.h"

namespace py = pybind11;

namespace {

struct MatrixView {
    const float* data;
    py::ssize_t rows;
    py::ssize_t cols;
    py::ssize_t row_stride;
    py::ssize_t col_stride;
};

void diff_1d(const float* first,
             py::ssize_t first_stride,
             const float* second,
             py::ssize_t second_stride,
             py::ssize_t len,
             float* out,
             py::ssize_t out_stride) {
    for (py::ssize_t i = 0; i < len; ++i) {
        out[i * out_stride] = second[i * second_stride] - first[i * first_stride];
    }
}

void compute_macd_matrix_by_column(const MatrixView& first,
                                   const MatrixView& second,
                                   float* out) {
    if (first.rows != second.rows || first.cols != second.cols) {
        throw std::runtime_error("EMA outputs for MACD dataframe must be matching 2D arrays.");
    }

    const py::ssize_t rows = first.rows;
    const py::ssize_t cols = first.cols;
    const size_t col_count = static_cast<size_t>(cols);
    const unsigned int hw = std::max(1u, std::thread::hardware_concurrency());
    const size_t num_threads = (col_count >= 32) ? std::min<size_t>(hw, col_count) : 1;
    const size_t chunk = (col_count + num_threads - 1) / num_threads;

    auto worker = [&](size_t start_col, size_t end_col) {
        for (size_t c = start_col; c < end_col; ++c) {
            diff_1d(first.data + static_cast<py::ssize_t>(c) * first.col_stride,
                    first.row_stride,
                    second.data + static_cast<py::ssize_t>(c) * second.col_stride,
                    second.row_stride,
                    rows,
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

py::object macd_series(py::object series, int window1, int window2) {
    py::array ema_w1 = ema_series(series, window1).attr("to_numpy")(py::arg("dtype") = py::dtype::of<float>(),
                                                                    py::arg("copy") = false);
    py::array ema_w2 = ema_series(series, window2).attr("to_numpy")(py::arg("dtype") = py::dtype::of<float>(),
                                                                    py::arg("copy") = false);

    auto w1_info = ema_w1.request();
    auto w2_info = ema_w2.request();
    if (w1_info.ndim != 1 || w2_info.ndim != 1 || w1_info.shape[0] != w2_info.shape[0]) {
        throw std::runtime_error("EMA outputs for MACD series must be matching 1D arrays.");
    }

    py::array_t<float> output(w1_info.shape[0]);
    auto out_info = output.request();

    {
        py::gil_scoped_release release_gil;
        diff_1d(static_cast<const float*>(w1_info.ptr),
                w1_info.strides[0] / static_cast<py::ssize_t>(sizeof(float)),
                static_cast<const float*>(w2_info.ptr),
                w2_info.strides[0] / static_cast<py::ssize_t>(sizeof(float)),
                w1_info.shape[0],
                static_cast<float*>(out_info.ptr),
                1);
    }

    return pandas_module().attr("Series")(output,
                                          py::arg("index") = series.attr("index"),
                                          py::arg("name") = series.attr("name"));
}

py::object macd_dataframe(py::object data_frame, int window1, int window2) {
    py::array ema_w1 = ema_dataframe(data_frame, window1).attr("to_numpy")(py::arg("dtype") = py::dtype::of<float>(),
                                                                            py::arg("copy") = false);
    py::array ema_w2 = ema_dataframe(data_frame, window2).attr("to_numpy")(py::arg("dtype") = py::dtype::of<float>(),
                                                                            py::arg("copy") = false);

    auto w1_info = ema_w1.request();
    auto w2_info = ema_w2.request();
    if (w1_info.ndim != 2 || w2_info.ndim != 2 ||
        w1_info.shape[0] != w2_info.shape[0] || w1_info.shape[1] != w2_info.shape[1]) {
        throw std::runtime_error("EMA outputs for MACD dataframe must be matching 2D arrays.");
    }

    py::array_t<float> output({w1_info.shape[0], w1_info.shape[1]});
    auto out_info = output.request();

    {
        py::gil_scoped_release release_gil;
        compute_macd_matrix_by_column(
            MatrixView{
                static_cast<const float*>(w1_info.ptr),
                w1_info.shape[0],
                w1_info.shape[1],
                w1_info.strides[0] / static_cast<py::ssize_t>(sizeof(float)),
                w1_info.strides[1] / static_cast<py::ssize_t>(sizeof(float)),
            },
            MatrixView{
                static_cast<const float*>(w2_info.ptr),
                w2_info.shape[0],
                w2_info.shape[1],
                w2_info.strides[0] / static_cast<py::ssize_t>(sizeof(float)),
                w2_info.strides[1] / static_cast<py::ssize_t>(sizeof(float)),
            },
            static_cast<float*>(out_info.ptr));
    }

    return pandas_module().attr("DataFrame")(output,
                                             py::arg("index") = data_frame.attr("index"),
                                             py::arg("columns") = data_frame.attr("columns"));
}

void register_macd(py::module_& m) {
    m.def("macd", &macd_series, py::arg("series"), py::arg("window1"), py::arg("window2"));
    m.def("macd_dataframe", &macd_dataframe, py::arg("dataframe"), py::arg("window1"), py::arg("window2"));
}
