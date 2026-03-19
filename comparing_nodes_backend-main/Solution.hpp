//
// Created by Alexandre Chanson on 27/11/2025.
//

#ifndef COMPARING_NODES_BACKEND_SOLUTION_HPP
#define COMPARING_NODES_BACKEND_SOLUTION_HPP

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <stdexcept>
#include "csv_loader.hpp"

class Solution {
public :
    //Solution structure : vector of len |indicators| :
    //  0 unused (default for partial solution / 1 used for comparison / - 1 used for clustering
    std::vector<short> space;
    csv::CSVNumericData& csv; // pointer to data source
    int nb_dims(){return space.size();}; // size of subspace

    // clustering related
    bool clusters_computed = false;
    int nb_clusters = -1;
    std::vector<int> membership;

    void reset_clustering_state() {
        clusters_computed = false;
        nb_clusters = -1;
        membership.clear();
    }

    // Constructors
    Solution(csv::CSVNumericData& data) : csv(data) {}
    explicit Solution(const std::string& pattern, csv::CSVNumericData& data);
    explicit Solution(std::size_t n, csv::CSVNumericData& data);
    Solution(const Solution& other, std::size_t index_to_change, short value_to_set);

    // Copy constructor
    Solution(const Solution& other) = default;

    // Copy assignment operator
    Solution& operator=(const Solution& other) {
        if (this != &other) {
            space = other.space;
            csv = other.csv;
            clusters_computed = other.clusters_computed;
            nb_clusters = other.nb_clusters;
            membership = other.membership;
        }
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const Solution& s);

    bool feasible() const {
        for (const auto i : space) {
            if (i == 1) {
                for (const auto j : space) {
                    if (j == -1) return true;
                }
            }
        }
        return false;
    }

    template<short FilterValue>
    class FilterIndexIterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type        = std::size_t;
        using difference_type   = std::ptrdiff_t;
        using pointer           = void;         // not used
        using reference         = std::size_t;  // returned by value

        FilterIndexIterator(const std::vector<short>* s, std::size_t i)
            : space_(s), idx_(i)
        {
            skip_to_next_valid();
        }

        value_type operator*() const {
            return idx_;
        }

        FilterIndexIterator& operator++() {
            ++idx_;
            skip_to_next_valid();
            return *this;
        }

        FilterIndexIterator operator++(int) {
            FilterIndexIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const FilterIndexIterator& other) const {
            return space_ == other.space_ && idx_ == other.idx_;
        }

        bool operator!=(const FilterIndexIterator& other) const {
            return !(*this == other);
        }

    private:
        const std::vector<short>* space_;
        std::size_t idx_;

        void skip_to_next_valid() {
            if (!space_) return;
            auto n = space_->size();
            while (idx_ < n && (*space_)[idx_] != FilterValue) {
                ++idx_;
            }
        }
    };

    void debug_print() {
        std::cout << "feasible ? " << feasible() << "\n";
        std::cout << "silhouette " << silhouette_score() << "\n";
        std::cout << "clustering :" << "\n";
        for (auto i : membership) {
            std::cout << i << ",";
        }
        std::cout  << "\n";
        std::cout << "obj  " << obj() << "\n";
    }

    template<short FilterValue>
    struct FilterIndexRange {
        const std::vector<short>* space;
        FilterIndexIterator<FilterValue> begin() const {
            return FilterIndexIterator<FilterValue>(space, 0);
        }
        FilterIndexIterator<FilterValue> end() const {
            return FilterIndexIterator<FilterValue>(space, space ? space->size() : 0);
        }
    };

    [[nodiscard]] FilterIndexRange<1> comp_idx() const {
        return FilterIndexRange<1>{ &space };
    }

    [[nodiscard]] FilterIndexRange<-1> clust_idx() const {
        return FilterIndexRange<-1>{ &space };
    }

    size_t nb_dims_comp() const {
        size_t nb = 0;
        for (std::size_t i = 0; i < space.size(); ++i) {
            if (space[i] == 1) ++nb;
        }
        return nb;
    }

    size_t nb_dims_clust() const {
        size_t nb = 0;
        for (std::size_t i = 0; i < space.size(); ++i) {
            if (space[i] == -1) ++nb;
        }
        return nb;
    }

    float silhouette_score() const
    {
        const std::size_t n = csv.rows;
        if (n == 0) {
            throw std::invalid_argument("silhouette_score: empty data");
        }
        if (membership.size() != n) {
            throw std::invalid_argument("silhouette_score: membership size != number of rows");
        }
        if (nb_clusters <= 1) {
            std::cerr << "error please input nb cluster" << std::endl;
            return 0.0; // not really defined, just return 0
        }

        // at least one column ?
        bool has_col = false;
        for (auto j : clust_idx()) { (void)j; has_col = true; break; }
        if (!has_col) {
            return 0.0; // no features → all distances 0 → silhouette not meaningful
        }

        // 1) Count points per cluster (ignore membership == -1)
        std::vector<int> cluster_sizes(nb_clusters, 0);
        for (std::size_t i = 0; i < n; ++i) {
            int c = membership[i];
            if (c >= 0 && c < nb_clusters) {
                ++cluster_sizes[c];
            }
        }

        float total_s = 0.0;
        std::size_t counted_points = 0;

        // 2) For each point, compute a(i) and b(i)
        for (std::size_t i = 0; i < n; ++i) {
            int ci = membership[i];
            if (ci < 0 || ci >= nb_clusters) {
                continue; // skip unassigned
            }

            int size_ci = cluster_sizes[ci];
            if (size_ci <= 1) {
                // singleton cluster silhouette usually taken as 0/ignored
                continue;
            }

            std::vector<float> sum_d(nb_clusters, 0.0);

            // accumulate distances to all other points using only selected columns
            for (std::size_t k = 0; k < n; ++k) {
                if (k == i) continue;
                int ck = membership[k];
                if (ck < 0 || ck >= nb_clusters) continue;

                float d = euclidean_distance(i, k, clust_idx());
                sum_d[ck] += d;
            }

            // a(i): avg distance to own cluster
            float a = sum_d[ci] / static_cast<float>(size_ci - 1);

            // b(i): best (lowest) avg distance to any other cluster
            float b = 2e32f;
            for (int c = 0; c < nb_clusters; ++c) {
                if (c == ci) continue;
                int size_c = cluster_sizes[c];
                if (size_c == 0) continue;

                float avg = sum_d[c] / static_cast<float>(size_c);
                if (avg < b) b = avg;
            }

            if (!std::isfinite(b)) {
                continue; // no other clusters with points
            }

            float s_i;
            if (a == 0.0 && b == 0.0) {
                s_i = 0.0;
            } else {
                s_i = (b - a) / std::max(a, b);
            }

            total_s += s_i;
            ++counted_points;
        }

        if (counted_points == 0) {
            return 0.0;
        }
        return total_s / static_cast<float>(counted_points);
    }

template<typename IndexRange>
float euclidean_distance(size_t i, size_t j, const IndexRange& cols) const
{
    float sum = 0.0;
    for (auto col : cols) {
        float diff = csv.data[i * csv.cols + col] - csv.data[j * csv.cols + col];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

    float obj() {
        float sum = 0.0;
        for (std::size_t i = 0; i < csv.rows; ++i) {
            for (std::size_t j = i + 1; j < csv.rows; ++j) {
                if (membership[i] == membership[j]) {
                    float comp = 0;
                    for (const auto col : comp_idx())
                        comp += abs(csv.data[i * csv.cols + col] - csv.data[j * csv.cols + col]);
                    float clust = 0;
                    for (const auto col : clust_idx())
                        clust += pow(csv.data[i * csv.cols + col] - csv.data[j * csv.cols + col], 2);
                    sum += (comp - clust);
                }
            }
        }
        return sum;
    }

};


#endif //COMPARING_NODES_BACKEND_SOLUTION_HPP