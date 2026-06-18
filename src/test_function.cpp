#include<vector>
#include<iostream>
#include<cmath>
using namespace std;
vector<float> rolling_rsi(const vector<float>& data,float window)
{
    float average_gain=0.0; 
    float average_loss=0.0;     
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
     
    if(average_loss==0.0)
    {
         
        rsi[int(window)]=100.0;
    }
    else
    {
        float RS=average_gain/average_loss;
        rsi[int(window)]=100.0-(100.0/(1.0+RS));
    }
    for (int i = window + 1; i < data.size(); ++i) {
        float change = data[i] - data[i - 1];
        float gain = (change > 0) ? change : 0.0;
        float loss = (change < 0) ? -change : 0.0;

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
int main()
{
    vector<float> data={1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    vector<float> t=rolling_rsi(data,5);
    for(auto i:t)
    {
        cout<<i<<endl;
    }
                  
}
