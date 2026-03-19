//
// Created by Alexandre Chanson on 27/11/2025.
//

#include "Solution.hpp"
#include "rng_freezer.hpp"

Solution::Solution(const std::string &pattern, csv::CSVNumericData& data) : csv(data) {
    std::istringstream iss(pattern);
    int val;
    while (iss >> val) {
        if (val < -1 || val > 1) {
            throw std::runtime_error("Invalid value in pattern (must be -1, 0, or 1)");
        }
        space.push_back(static_cast<short>(val));
    }
    if (space.empty()) {
        throw std::runtime_error("Pattern string produced an empty solution");
    }
    reset_clustering_state();
}

std::ostream& operator<<(std::ostream& os, const Solution& s) {
    os << "[";
    for (std::size_t i = 0; i < s.space.size(); ++i) {
        if (i > 0) os << ",";
        os << s.space[i];
    }
    os << "]";
    return os;
}

Solution::Solution(std::size_t n, csv::CSVNumericData& data) : csv(data) {
    if (n < 2) {
        throw std::invalid_argument("Need size >= 2 to guarantee at least one 1 and one -1");
    }

    space.assign(n, 0);

    std::random_device rd;
    std::mt19937 local_gen(rd());
    std::mt19937& gen = rng_freezer::freeze_rng ? rng_freezer::frozen_engine : local_gen;

    // choose two distinct indices for guaranteed 1 and -1
    std::uniform_int_distribution<std::size_t> idx_dist(0, n - 1);
    std::size_t pos_pos = idx_dist(gen);
    std::size_t pos_neg;
    do {
        pos_neg = idx_dist(gen);
    } while (pos_neg == pos_pos);

    space[pos_pos] = 1;
    space[pos_neg] = -1;

    // fill remaining entries randomly with -1, 0, or 1
    std::uniform_int_distribution<int> val_dist(-1, 1);
    for (std::size_t i = 0; i < n; ++i) {
        if (i == pos_pos || i == pos_neg) continue;
        space[i] = static_cast<short>(val_dist(gen));
    }
    reset_clustering_state();
}

Solution::Solution(const Solution& other, std::size_t index_to_change, short value_to_set) : space(other.space),
    csv(other.csv) {
    if (index_to_change >= space.size()) {
        throw std::out_of_range("index_to_flip out of range");
    }

    space[index_to_change] = value_to_set;
    reset_clustering_state();
}