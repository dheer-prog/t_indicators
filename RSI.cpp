#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
using namespace std;
namespace py=pybind11;
py::array_t<float> rolling_rsi(py::array_t<float> data,int window)
{
    py::array_t<float> rsi({data.size()});
    py::buffer_info rsi_buf = rsi.request();
    float* rsi_ptr = (float*) rsi_buf.ptr;
    for(size_t i=0; i<data.size(); ++i) rsi_ptr[i] = NAN;
    py::buffer_info data_buf = data.request();
    float* data_ptr = (float*) data_buf.ptr;
    float average_gain=0.0f; 
    float average_loss=0.0f;     
     
    if(window>=(int)data.size())
    {
        return rsi;
    }
    for(int i=1;i<=window;i++)
    {
        float change=data_ptr[i]-data_ptr[i-1];
        if(change>0)
        {
            average_gain=average_gain+change;

        }
        else
        {
            average_loss=average_loss-change;
        }
        
       
    }
    average_gain=average_gain/window;
    average_loss=average_loss/window; 
    if(average_loss==0.0f)
    {
        rsi_ptr[window]=100.0f;
    }
    else
    {
        float RS=average_gain/average_loss;
        rsi_ptr[window]=100.0f-(100.0f/(1.0f+RS));
    }
    for (int i = window + 1; i < (int)data.size(); ++i) {
        float change = data_ptr[i] - data_ptr[i - 1];
        float gain = (change > 0) ? change : 0.0f;
        float loss = (change < 0) ? -change : 0.0f;

        // Wilder’s EMA smoothing
        average_gain = (average_gain* (window - 1) + gain) / window;
        average_loss = (average_loss* (window - 1) + loss) / window;

        if (average_loss == 0.0f)
            rsi_ptr[i] = 100.0f;
        else {
            float rs = average_gain / average_loss;
            rsi_ptr[i] = 100.0f - (100.0f / (1.0f + rs));
    }
    }

    return rsi;
}
py::object rsi_series(py::object series, int window)
{
    py::module_ np=py::module::import("numpy");
    py::module_ pd=py::module::import("pandas");
    py::array_t<float> values = np.attr("asarray")(series).cast<py::array_t<float>>();
    py::object index=series.attr("index");
    py::array_t<float> rsi=rolling_rsi(values,window);
    return pd.attr("Series")(rsi,py::arg("index")=index);

}
py::object rsi_dataframe(py::object dataframe,int window)
{
    py::module_ pd=py::module_::import("pandas");
    py::dict new_df;
    py::object cols=dataframe.attr("columns");
    for(py::handle col: cols)
    {
        py::object series_col=dataframe.attr("__getitem__")(col);
        py::object rsi_series_result=rsi_series(series_col,window);
        new_df[col]=rsi_series_result;

    }
    return pd.attr("DataFrame")(new_df, py::arg("index")=dataframe.attr("index"));
}
void register_rsi(py::module_ &m)
{
    m.def("RSI", &rsi_series,"Computing RSI of series ");
    m.def("RSI_DataFrame",&rsi_dataframe,"Calculate RSI of entire dataframe");

}

    

 