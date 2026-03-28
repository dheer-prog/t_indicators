#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>
namespace py=pybind11;

void register_rolling(py::module_ &m);
void register_rsi(py::module_ &m);
void register_williams_r(py::module_ &m);
void register_trend_ewma(py::module_ &m);
void register_volatility(py::module_ &m);
PYBIND11_MODULE(t_indicators, m)
{
    m.doc()="Technical Indicators caculation module";
    register_rolling(m);
    register_rsi(m);
    register_williams_r(m);
    register_trend_ewma(m);
    register_volatility(m);
}


//You need to fix trend's code