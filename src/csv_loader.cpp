# include "include/csv.h"

int main(){
  io::CSVReader<3> in("/home/dhher/t_indicators/data.csv");
  in.read_header(io::ignore_extra_column, "date", "AAPL", "ADBE");
  std::string date; double aapl; double adbe;
  while(in.read_row(date, aapl, adbe)){
        
  }
}