#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <thread>
#include <vector>
#include "helper.h"
using namespace std;
namespace py=pybind11;
namespace{
    struct MatrixView {
        const float* data;
        py::ssize_t rows;
        py::ssize_t cols;
        py::ssize_t row_stride;
        py::ssize_t col_stride;
    };

//In this entire code the fundamental assumption is out_stride and in_stride is 1

void compute_rsi_1d(const float* data,
                    py::ssize_t len,
                    py::ssize_t in_stride,
                    float* out,
                    py::ssize_t window=14,
                    py::ssize_t out_stride=1){
                
    if (window == 0 || window >= len) {
        return;
    }
    float gain=0.0f;
    float loss=0.0f; 
    float init_price=data[0]; 
    for(py::ssize_t i=1;i<=window;i++){
        if(init_price>data[i * in_stride]){
            loss=loss+abs(init_price-data[i * in_stride]);
        }
        else{
            gain=gain+data[i * in_stride]-init_price; 
        }
        init_price=data[i*in_stride]; 
    } 
    float wf=static_cast<float>(window); 
    float avg_gain=gain/wf; 
    float avg_loss=loss/wf;
    float rs_init;
    if(avg_loss==0){
        rs_init=inf_val(); 
    }
    else{
        rs_init=1+(avg_gain/avg_loss);
    }
    out[window * out_stride]=100*(1.0f-(1/rs_init));
    float current_gain=0.0f; 
    float current_loss=0.0f;  
    for(py::ssize_t i=window+1;i<len;i++){
        if(data[i * in_stride]>data[(i-1) * in_stride]){
            current_gain=data[i * in_stride]-data[(i-1) * in_stride];
            current_loss=0; 
        }
        else{
            current_loss=data[(i-1) * in_stride]-data[i * in_stride]; 
            current_gain=0; 
        }
        avg_gain=((avg_gain*(window-1))+current_gain)/wf;
        avg_loss=((avg_loss*(window-1))+current_loss)/wf;
        rs_init=1+(avg_gain/avg_loss);  
        out[i * out_stride]=100*(1-(1/rs_init)); 
    }
}
void compute_rsi_matrix_by_column(const MatrixView& input,size_t window, float* out){
    const py::ssize_t rows = input.rows;
    const py::ssize_t cols = input.cols;
    const size_t col_count = static_cast<size_t>(cols);
    std::fill_n(out, rows * cols, nan_value());
    if (window == 0 || static_cast<py::ssize_t>(window) > rows) {
        return;
    }
     
    const unsigned int hw = std::max(1u, std::thread::hardware_concurrency());
    const size_t num_threads = (col_count >= 32) ? std::min<size_t>(hw, col_count) : 1;
    const size_t chunk = (col_count + num_threads - 1) / num_threads;
    auto worker = [&](size_t start_col, size_t end_col) {
        for (size_t c = start_col; c < end_col; ++c) {
            compute_rsi_1d(input.data + static_cast<py::ssize_t>(c) * input.col_stride,
                           rows,
                           input.row_stride,
                           out + static_cast<py::ssize_t>(c),
                           static_cast<py::ssize_t>(window),
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
}

py::object rsi_series(py::object series, int window)
{
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
        compute_rsi_1d(static_cast<const float*>(info.ptr),
                       info.shape[0],
                       info.strides[0] / static_cast<py::ssize_t>(sizeof(float)),
                       static_cast<float*>(out_info.ptr),
                       static_cast<py::ssize_t>(window),
                       1);
    }

    return pandas_module().attr("Series")(output,
                                          py::arg("index") = series.attr("index"),
                                          py::arg("name") = series.attr("name"));


}
py::object rsi_dataframe(py::object data_frame, int window) {
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
        compute_rsi_matrix_by_column(view, static_cast<size_t>(window), static_cast<float*>(out_info.ptr));
    }

    return pandas_module().attr("DataFrame")(output,
                                             py::arg("index") = data_frame.attr("index"),
                                             py::arg("columns") = data_frame.attr("columns"));
}
void register_rsi(py::module_ &m)
{
    m.def("RSI", &rsi_series,"Computing RSI of series ");
    m.def("RSI_DataFrame",&rsi_dataframe,"Calculate RSI of entire dataframe");

}

    

 
