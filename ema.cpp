#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
using namespace std;
namespace py=pybind11;
py::array_t<float> rolling_ema(py::array_t<float> data,int window)
{
    py::array_t<float>ema({data.size()});
    py::buffer_info ema_buf=ema.request();
    float* ema_ptr=(float*)ema_buf.ptr;
    for(size_t i=0; i<data.size(); ++i) ema_ptr[i]=NAN; 
    py::buffer_info data_buf=data.request();
    float* data_ptr=(float*)data_buf.ptr;
    float smoothing_factor=2.0f/(window+1);
    float sum=0.0f;
    for(int i=0;i<window;i++){
        sum=sum+data_ptr[i];
    }
    ema_ptr[window-1]=sum/window;
    for(int i=window;i<(int)data.size();i++){
        ema_ptr[i]=(smoothing_factor*data_ptr[i])+(1-smoothing_factor)*ema_ptr[i-1];
    }
    return ema;
}
py:: object ema_series(py::object series,int window)
{
    py::module_ np = py::module_::import("numpy");
    py::module_ pd = py::module_::import("pandas");
    py::array_t<float> values=np.attr("asarray")(series).cast<py::array_t<float>>();
    py::object index=series.attr("index");

    py::array_t<float> result=rolling_ema(values,window);
    return pd.attr("Series")(result,py::arg("index")=index);
}
py::object ema_dataframe(py::object dataframe,int window)
{
    py::module_ pd = py::module_::import("pandas");
    py::dict new_df;
    py::object index=dataframe.attr("index");
    py::object cols =dataframe.attr("columns");
    for(py::handle col: cols)
    {
        py::object series_vals=dataframe.attr("__getitem__")(col);
        py::object result=rolling_ema(series_vals,window);
        new_df[col]=result;
    }
    return pd.attr("DataFrame")(new_df,py::arg("index")=index);
}
void register_ema(py::module_ &m)
{
    m.def("ema_series",&ema_series,py::arg("series"),py::arg("window"));
    m.def("ema",&ema_dataframe,py::arg("dataframe"),py::arg("window"));
}
