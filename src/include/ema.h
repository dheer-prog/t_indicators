#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

py::object ema_series(py::object series, int window);
py::object ema_dataframe(py::object data_frame, int window);
void register_ema(py::module_& m);
