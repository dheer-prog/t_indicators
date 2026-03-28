#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
using namespace std;
namespace py=pybind11;
vector<float> rolling_rsi(const vector<float>& data,int window)
{
    float average_gain=0.0f; 
    float average_loss=0.0f;     
    vector<float>rsi(data.size(),NAN);
     
    if(window>=data.size())
    {
        return rsi;
    }
    for(int i=1;i<=window;i++)
    {
        float change=data[i]-data[i-1];
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
        rsi[window]=100.0f;
    }
    else
    {
        float RS=average_gain/average_loss;
        rsi[window]=100.0f-(100.0f/(1.0f+RS));
    }
    for (int i = window + 1; i < data.size(); ++i) {
        float change = data[i] - data[i - 1];
        float gain = (change > 0) ? change : 0.0f;
        float loss = (change < 0) ? -change : 0.0f;

        // Wilder’s EMA smoothing
        average_gain = (average_gain* (window - 1) + gain) / window;
        average_loss = (average_loss* (window - 1) + loss) / window;

        if (average_loss == 0.0f)
            rsi[i] = 100.0f;
        else {
            float rs = average_gain / average_loss;
            rsi[i] = 100.0f - (100.0f / (1.0f + rs));
    }
    }

    return rsi;
}
py::object rsi_series(py::object series, int window)
{
    py::module_ np=py::module::import("numpy");
    py::module_ pd=py::module::import("pandas");
    auto values=np.attr("asarray")(series).cast<vector<float>>(); 
    auto index=series.attr("index");
    auto rsi=rolling_rsi(values,window);
    py::array result_array=py::cast(rsi);

    return pd.attr("Series")(result_array,py::arg("index")=index);

}
py::object rsi_dataframe(py::object dataframe,int window)
{
    py::module_ pd=py::module_::import("pandas");
    py::dict new_df;
    auto cols=dataframe.attr("columns");
    for(auto col:cols)
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

    

 