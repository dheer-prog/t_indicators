#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct CsvLayout {
    std::vector<std::string> header;
    std::size_t total_rows = 0;
    std::uintmax_t file_size_bytes = 0;
};

std::string trim_copy(const std::string& value) {
    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])) != 0) {
        ++start;
    }

    std::size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }

    return value.substr(start, end - start);
}

std::string strip_trailing_cr(std::string line) {
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    return line;
}

std::vector<std::string> make_default_header(std::size_t column_count) {
    std::vector<std::string> header;
    header.reserve(column_count);
    for (std::size_t index = 0; index < column_count; ++index) {
        header.push_back("column_" + std::to_string(index));
    }
    return header;
}

std::vector<std::string> parse_csv_line(const std::string& line, char delimiter) {
    std::vector<std::string> fields;
    std::string current;
    bool in_quotes = false;

    for (std::size_t index = 0; index < line.size(); ++index) {
        const char ch = line[index];

        if (ch == '"') {
            if (in_quotes && index + 1 < line.size() && line[index + 1] == '"') {
                current.push_back('"');
                ++index;
            } else {
                in_quotes = !in_quotes;
            }
            continue;
        }

        if (ch == delimiter && !in_quotes) {
            fields.push_back(current);
            current.clear();
            continue;
        }

        current.push_back(ch);
    }

    fields.push_back(current);
    return fields;
}

double parse_numeric_or_nan(const std::string& value) {
    const std::string trimmed = trim_copy(value);
    if (trimmed.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    errno = 0;
    char* end = nullptr;
    const double parsed = std::strtod(trimmed.c_str(), &end);
    if (end == trimmed.c_str() || *end != '\0' || errno == ERANGE) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return parsed;
}

char resolve_delimiter(const std::string& delimiter) {
    if (delimiter.size() != 1) {
        throw std::runtime_error("delimiter must be a single character");
    }
    return delimiter[0];
}

CsvLayout inspect_csv_layout(const std::string& path, char delimiter, bool has_header) {
    CsvLayout layout;
    layout.file_size_bytes = std::filesystem::file_size(path);

    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open CSV file: " + path);
    }

    std::string line;
    if (!std::getline(file, line)) {
        return layout;
    }

    line = strip_trailing_cr(std::move(line));
    std::vector<std::string> first_row = parse_csv_line(line, delimiter);

    if (has_header) {
        layout.header = std::move(first_row);
    } else {
        layout.header = make_default_header(first_row.size());
        layout.total_rows = 1;
    }

    while (std::getline(file, line)) {
        ++layout.total_rows;
    }

    return layout;
}

std::size_t find_column_index(const std::vector<std::string>& header, const std::string& column_name) {
    const auto iterator = std::find(header.begin(), header.end(), column_name);
    if (iterator == header.end()) {
        throw std::runtime_error("Column not found: " + column_name);
    }
    return static_cast<std::size_t>(std::distance(header.begin(), iterator));
}

py::dict inspect_csv(const std::string& path,
                     bool has_header = true,
                     const std::string& delimiter = ",") {
    const char delimiter_char = resolve_delimiter(delimiter);
    const CsvLayout layout = inspect_csv_layout(path, delimiter_char, has_header);

    py::dict result;
    result["path"] = path;
    result["has_header"] = has_header;
    result["delimiter"] = delimiter;
    result["columns"] = layout.header;
    result["column_count"] = py::int_(layout.header.size());
    result["total_rows"] = py::int_(layout.total_rows);
    result["file_size_bytes"] = py::int_(layout.file_size_bytes);
    result["file_size_mb"] = py::float_(static_cast<double>(layout.file_size_bytes) / (1024.0 * 1024.0));
    return result;
}

py::dict load_csv(const std::string& path,
                  py::ssize_t start_row = 0,
                  py::ssize_t max_rows = -1,
                  const std::vector<std::string>& columns = {},
                  const std::string& index_column = "",
                  bool has_header = true,
                  const std::string& delimiter = ",",
                  std::size_t auto_limit_mb = 128,
                  std::size_t large_file_rows = 50000) {
    if (start_row < 0) {
        throw std::runtime_error("start_row must be >= 0");
    }

    if (max_rows < -1) {
        throw std::runtime_error("max_rows must be -1 or >= 0");
    }

    const char delimiter_char = resolve_delimiter(delimiter);
    const CsvLayout layout = inspect_csv_layout(path, delimiter_char, has_header);

    std::size_t effective_max_rows = 0;
    bool auto_limited = false;
    if (max_rows >= 0) {
        effective_max_rows = static_cast<std::size_t>(max_rows);
    } else if (layout.file_size_bytes > auto_limit_mb * 1024ULL * 1024ULL) {
        effective_max_rows = large_file_rows;
        auto_limited = true;
    } else {
        effective_max_rows = std::numeric_limits<std::size_t>::max();
    }

    std::size_t index_column_idx = std::numeric_limits<std::size_t>::max();
    if (!index_column.empty()) {
        index_column_idx = find_column_index(layout.header, index_column);
    }

    std::vector<std::size_t> selected_indices;
    std::vector<std::string> selected_columns;

    if (columns.empty()) {
        selected_indices.reserve(layout.header.size());
        selected_columns.reserve(layout.header.size());
        for (std::size_t index = 0; index < layout.header.size(); ++index) {
            if (index == index_column_idx) {
                continue;
            }
            selected_indices.push_back(index);
            selected_columns.push_back(layout.header[index]);
        }
    } else {
        selected_indices.reserve(columns.size());
        selected_columns.reserve(columns.size());
        for (const std::string& column_name : columns) {
            if (!index_column.empty() && column_name == index_column) {
                throw std::runtime_error("index_column cannot also be requested as a data column");
            }
            const std::size_t index = find_column_index(layout.header, column_name);
            selected_indices.push_back(index);
            selected_columns.push_back(column_name);
        }
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open CSV file: " + path);
    }

    std::string line;
    std::vector<std::string> buffered_first_row;
    if (!std::getline(file, line)) {
        py::dict empty_result;
        empty_result["values"] = py::array_t<double>(
            std::vector<py::ssize_t>{0, static_cast<py::ssize_t>(selected_columns.size())});
        empty_result["columns"] = selected_columns;
        empty_result["index"] = py::list();
        empty_result["rows_loaded"] = py::int_(0);
        empty_result["start_row"] = py::int_(start_row);
        empty_result["next_row"] = py::none();
        empty_result["total_rows"] = py::int_(layout.total_rows);
        empty_result["truncated"] = py::bool_(false);
        empty_result["auto_limited"] = py::bool_(false);
        empty_result["file_size_bytes"] = py::int_(layout.file_size_bytes);
        return empty_result;
    }

    if (!has_header) {
        buffered_first_row = parse_csv_line(strip_trailing_cr(std::move(line)), delimiter_char);
    }

    std::vector<double> flat_values;
    std::vector<std::string> index_values;
    const std::size_t column_count = selected_indices.size();
    if (effective_max_rows != std::numeric_limits<std::size_t>::max()) {
        flat_values.reserve(effective_max_rows * column_count);
        if (index_column_idx != std::numeric_limits<std::size_t>::max()) {
            index_values.reserve(effective_max_rows);
        }
    }

    auto append_row = [&](const std::vector<std::string>& row_fields) {
        for (const std::size_t column_idx : selected_indices) {
            if (column_idx < row_fields.size()) {
                flat_values.push_back(parse_numeric_or_nan(row_fields[column_idx]));
            } else {
                flat_values.push_back(std::numeric_limits<double>::quiet_NaN());
            }
        }

        if (index_column_idx != std::numeric_limits<std::size_t>::max()) {
            if (index_column_idx < row_fields.size()) {
                index_values.push_back(row_fields[index_column_idx]);
            } else {
                index_values.emplace_back();
            }
        }
    };

    std::size_t current_row = 0;
    std::size_t rows_loaded = 0;

    auto maybe_take_row = [&](const std::vector<std::string>& row_fields) -> bool {
        if (current_row >= static_cast<std::size_t>(start_row)) {
            if (rows_loaded >= effective_max_rows) {
                return false;
            }
            append_row(row_fields);
            ++rows_loaded;
        }
        ++current_row;
        return true;
    };

    if (!has_header) {
        maybe_take_row(buffered_first_row);
    }

    while (std::getline(file, line)) {
        const std::vector<std::string> row_fields = parse_csv_line(strip_trailing_cr(std::move(line)), delimiter_char);
        if (!maybe_take_row(row_fields)) {
            break;
        }
    }

    py::array_t<double> values({static_cast<py::ssize_t>(rows_loaded), static_cast<py::ssize_t>(column_count)});
    auto values_buffer = values.mutable_unchecked<2>();
    for (std::size_t row = 0; row < rows_loaded; ++row) {
        for (std::size_t col = 0; col < column_count; ++col) {
            values_buffer(static_cast<py::ssize_t>(row), static_cast<py::ssize_t>(col)) =
                flat_values[row * column_count + col];
        }
    }

    py::dict result;
    result["values"] = values;
    result["columns"] = selected_columns;
    result["index"] = py::cast(index_values);
    result["rows_loaded"] = py::int_(rows_loaded);
    result["start_row"] = py::int_(start_row);
    result["total_rows"] = py::int_(layout.total_rows);
    result["file_size_bytes"] = py::int_(layout.file_size_bytes);
    result["auto_limited"] = py::bool_(auto_limited);

    const bool truncated = (static_cast<std::size_t>(start_row) + rows_loaded) < layout.total_rows;
    result["truncated"] = py::bool_(truncated);
    if (truncated) {
        result["next_row"] = py::int_(static_cast<std::size_t>(start_row) + rows_loaded);
    } else {
        result["next_row"] = py::none();
    }

    return result;
}

}  // namespace

void register_csv(py::module_& m) {
    m.def("inspect_csv",
          &inspect_csv,
          py::arg("path"),
          py::arg("has_header") = true,
          py::arg("delimiter") = ",",
          "Inspect a CSV file without loading it into memory.");

    m.def("load_csv",
          &load_csv,
          py::arg("path"),
          py::arg("start_row") = 0,
          py::arg("max_rows") = -1,
          py::arg("columns") = std::vector<std::string>{},
          py::arg("index_column") = "",
          py::arg("has_header") = true,
          py::arg("delimiter") = ",",
          py::arg("auto_limit_mb") = 128,
          py::arg("large_file_rows") = 50000,
          "Stream a CSV file and load only the requested rows and columns.");
}
