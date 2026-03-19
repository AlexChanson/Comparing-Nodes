
#ifndef COMPARING_NODES_BACKEND_CSV_LOADER_HPP
#define COMPARING_NODES_BACKEND_CSV_LOADER_HPP
#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

namespace csv {

struct CSVNumericData {
    std::vector<std::string> headers;           // names of numeric columns
    std::vector<float> data;      // data[row * cols + col]
    size_t rows = 0;
    size_t cols = 0;
    std::vector<size_t> original_col_indices;   // original indices of kept columns
};

// ---- Helper functions (inline for header-only use) ----

inline std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

inline std::vector<std::string> split_line(const std::string& line, char delim) {
    std::vector<std::string> tokens;
    std::string token;
    std::stringstream ss(line);
    while (std::getline(ss, token, delim)) {
        tokens.push_back(trim(token));
    }
    return tokens;
}

inline bool is_number(const std::string& s) {
    if (s.empty()) return false;   // treat empty as non-numeric; change if needed
    try {
        size_t pos;
        std::stod(s, &pos);
        return pos == s.size();    // no trailing junk
    } catch (...) {
        return false;
    }
}

// ---- Main loader ----

inline CSVNumericData load_numeric_csv(const std::string& filename, char delimiter = ',') {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;

    // 1. Read header
    if (!std::getline(file, line)) {
        throw std::runtime_error("Empty file or cannot read header");
    }
    std::vector<std::string> all_headers = split_line(line, delimiter);
    const size_t num_cols = all_headers.size();

    // 2. Read all rows as strings
    std::vector<std::vector<std::string>> raw_rows;
    while (std::getline(file, line)) {
        if (line.empty()) continue; // skip empty lines
        std::vector<std::string> row = split_line(line, delimiter);

        if (row.size() < num_cols) {
            row.resize(num_cols, "");  // missing values -> empty
        } else if (row.size() > num_cols) {
            row.resize(num_cols);      // ignore extra columns
        }

        raw_rows.push_back(std::move(row));
    }

    const size_t num_rows = raw_rows.size();

    // 3. Determine which columns are fully numeric
    std::vector<bool> is_numeric_col(num_cols, true);

    for (size_t col = 0; col < num_cols; ++col) {
        for (size_t row = 0; row < num_rows; ++row) {
            const std::string& cell = raw_rows[row][col];
            if (!is_number(cell)) {
                is_numeric_col[col] = false;
                break;
            }
        }
    }

    // 4. Build result
    CSVNumericData result;
    std::vector<int> col_map(num_cols, -1);
    size_t numeric_count = 0;

    for (size_t col = 0; col < num_cols; ++col) {
        if (is_numeric_col[col]) {
            result.headers.push_back(all_headers[col]);
            result.original_col_indices.push_back(col);
            col_map[col] = static_cast<int>(numeric_count++);
        }
    }

    result.rows = num_rows;
    result.cols = numeric_count;
    result.data.resize(num_rows * numeric_count);

    for (size_t row = 0; row < num_rows; ++row) {
        for (size_t col = 0; col < num_cols; ++col) {
            if (!is_numeric_col[col]) continue;
            int dst_col = col_map[col];
            const std::string& cell = raw_rows[row][col];
            result.data[row * numeric_count + dst_col] = std::stod(cell);  // safe now
        }
    }

    return result;
}

} // namespace csv

#endif //COMPARING_NODES_BACKEND_CSV_LOADER_HPP