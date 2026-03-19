//
// Created by Alexandre Chanson on 28/11/2025.
//

#ifndef COMPARING_NODES_BACKEND_CUT_HPP
#define COMPARING_NODES_BACKEND_CUT_HPP
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include "rng_freezer.hpp"
#include "Solution.hpp"

class Cut {
public:
    using Membership = std::vector<int>;


    static Membership min_cut(Solution& sol, const int k, const float alpha, const float iter_factor = 5, const float temp_factor = 1.0, float* out_weight = nullptr) {

        const int n_iters = sol.csv.rows * iter_factor;
        float temperature = sol.csv.rows * temp_factor;
        std::size_t n = sol.csv.rows;

        // Setup Random Engine
        std::random_device rd;
        std::mt19937 local_gen(rd());
        std::mt19937& gen = rng_freezer::freeze_rng ? rng_freezer::frozen_engine : local_gen;

        std::uniform_int_distribution<int> group_dist(0, k - 1);
        std::uniform_int_distribution<int> index_dist(0, n - 1);
        std::uniform_real_distribution<float> real_dist(0.0, 1.0);

        // Initialize Random Solution
        Membership current_sol(n);
        std::generate(current_sol.begin(), current_sol.end(), [&]() {
            return group_dist(gen);
        });

        Membership best_sol = current_sol;

        // --- INITIAL FULL COMPUTATION ---
        // We separate the raw cut size from the penalty components for incremental updates
        float current_raw_cut = eval_cut_size(sol, current_sol, k);

        // Pre-calculate max distances for every cluster to maintain state
        std::vector<float> current_cluster_maxes(k, 0.0f);
        float current_penalty_sum = 0.0f;
        for(int i=0; i<k; ++i) {
            current_cluster_maxes[i] = compute_cluster_diameter(sol, current_sol, i);
            current_penalty_sum += current_cluster_maxes[i];
        }

        float current_val = current_raw_cut * current_penalty_sum;
        float best_val = current_val;

        // 3. Annealing Loop
        for (int i = 0; i < n_iters; ++i) {

            // A. Select Move
            int node_to_move = index_dist(gen);
            int current_group = current_sol[node_to_move];
            int new_group = current_group;

            while (new_group == current_group) {
                new_group = group_dist(gen);
            }

            // B. Calculate Energy Incrementally

            // Cut Size Delta
            float delta_raw = eval_cut_delta(sol, current_sol, node_to_move, current_group, new_group);
            float next_raw_cut = current_raw_cut + delta_raw;

            // Penalty Delta
            //    Old Group: Node leaves. We must recompute diameter excluding this node.
            float p_old_group_curr = current_cluster_maxes[current_group];
            float p_old_group_next = compute_cluster_diameter(sol, current_sol, current_group, node_to_move);

            //    New Group: Node enters. Max is max(current_max, max_dist_from_node_to_others).
            float p_new_group_curr = current_cluster_maxes[new_group];
            float dist_to_others = compute_node_cluster_max(sol, current_sol, node_to_move, new_group);
            float p_new_group_next = std::max(p_new_group_curr, dist_to_others);

            float next_penalty_sum = current_penalty_sum - p_old_group_curr - p_new_group_curr
                                                         + p_old_group_next + p_new_group_next;

            // Calculate total Value
            float next_val = next_raw_cut * next_penalty_sum;
            float delta = next_val - current_val;

            // C. Metropolis Criterion
            bool accept = false;
            if (delta > 0) {
                accept = true;
            } else {
                float probability = std::exp(delta / temperature);
                if (real_dist(gen) < probability) {
                    accept = true;
                }
            }

            if (accept) {
                // Apply changes to stored states
                current_sol[node_to_move] = new_group;
                current_raw_cut = next_raw_cut;

                current_cluster_maxes[current_group] = p_old_group_next;
                current_cluster_maxes[new_group]     = p_new_group_next;
                current_penalty_sum                  = next_penalty_sum;
                current_val                          = next_val;

                if (current_val > best_val) {
                    best_sol = current_sol;
                    best_val = current_val;
                }
            }
            // D. Cool down
            temperature *= alpha;
        }

        if (out_weight) {
            *out_weight = best_val;
        }
        return best_sol;
    }

    // Recompute diameter of one specific cluster
    // If skip_node_idx is provided (!= -1), that node is ignored (treated as removed)
    static float compute_cluster_diameter(Solution& sol, const Membership& m, int k_idx, int skip_node_idx = -1) {
        std::vector<size_t> indices;
        indices.reserve(m.size() / 2); // heuristic reserve

        for (size_t i = 0; i < m.size(); ++i) {
            if (m[i] == k_idx && (int)i != skip_node_idx) {
                indices.push_back(i);
            }
        }

        if (indices.empty()) return 0.0f;

        float local_max = 0;
        const size_t n_points = indices.size();

        #pragma omp parallel for reduction(max:local_max) schedule(runtime)
        for (size_t i = 0; i < n_points; ++i) {
            const size_t p1 = indices[i];
            for (size_t j = i + 1; j < n_points; ++j) {
                float d = 0;
                size_t p2 = indices[j];
                #pragma omp simd reduction(+:d)
                for (int dim = 0; dim < sol.nb_dims(); dim++) {
                    // Penalty uses space[dim] == -1
                    if(sol.space[dim] == -1) {
                        float diff = sol.csv.data[p1 * sol.csv.cols + dim] - sol.csv.data[p2 * sol.csv.cols + dim];
                        d += diff * diff;
                    }
                }
                if (d > local_max) {
                    local_max = d;
                }
            }
        }
        return local_max;
    }

    // Compute max distance from a specific node to a cluster
    // Used when a node enters a group
    static float compute_node_cluster_max(Solution& sol, const Membership& m, int node_idx, int k_idx) {
        float local_max = 0.0f;
        const size_t n_rows = sol.csv.rows;

        #pragma omp parallel for reduction(max:local_max) schedule(static)
        for(size_t i = 0; i < n_rows; ++i) {
            if(m[i] == k_idx && (int)i != node_idx) {
                float d = 0;
                #pragma omp simd reduction(+:d)
                for (int dim = 0; dim < sol.nb_dims(); dim++) {
                     // Penalty uses space[dim] == -1
                    if(sol.space[dim] == -1) {
                        float diff = sol.csv.data[node_idx * sol.csv.cols + dim] - sol.csv.data[i * sol.csv.cols + dim];
                        d += diff * diff;
                    }
                }
                if (d > local_max) local_max = d;
            }
        }
        return local_max;
    }

    // Standard penalty calculation
    static float cut_penalty(Solution& sol, const Membership& m, const int k) {
        std::vector<float> maxes(k, 0);
        for (int l = 0; l < k; ++l) {
            maxes[l] = compute_cluster_diameter(sol, m, l);
        }
        float sum = 0.0;
        for (float max_cls : maxes) {
            sum += max_cls;
        }
        return sum;
    }

    // Standard cut calculation
    static float eval_cut_size(Solution& sol, const Membership& m, const int k) {
    // Pre-compute active dimensions
    std::vector<int> active_dims;
    for (int dim = 0; dim < sol.nb_dims(); ++dim) {
        if (sol.space[dim] == 1) {
            active_dims.push_back(dim);
        }
    }

    // group indices by cluster
    std::vector<std::vector<size_t>> cluster_indices(k);
    for (size_t i = 0; i < m.size(); ++i) {
        if (m[i] >= 0 && m[i] < k) {
            cluster_indices[m[i]].push_back(i);
        }
    }

    float total_sum = 0.0;

    // Iterate only over pairs of DIFFERENT clusters, ie. crosses the cut
    for (int c1 = 0; c1 < k; ++c1) {
        for (int c2 = c1 + 1; c2 < k; ++c2) {

            const auto& group_a = cluster_indices[c1];
            const auto& group_b = cluster_indices[c2];

            if (group_a.empty() || group_b.empty()) continue;

            float cluster_pair_sum = 0.0;

            // Parallelize the work for this specific pair of clusters
            #pragma omp parallel for reduction(+:cluster_pair_sum) schedule(runtime)
            for (const auto idx_a : group_a) {
                for (const auto idx_b : group_b) {
                    float d = 0.0f;

                    // Vectorized loop over comparison dimensions only
                    #pragma omp simd reduction(+:d)
                    for (size_t i = 0; i < active_dims.size(); ++i) {
                        const int dim = active_dims[i];
                        d += std::fabsf(sol.csv.data[idx_a * sol.csv.cols + dim] - sol.csv.data[idx_b * sol.csv.cols + dim]);
                    }
                    cluster_pair_sum += d;
                }
            }
            total_sum += cluster_pair_sum;
        }
    }

    return total_sum;
}

    static float eval_cut_delta(Solution& sol, const Membership& m, int node_idx, int old_group, int new_group) {
        std::vector<int> active_dims;
        active_dims.reserve(sol.nb_dims());
        for (int dim = 0; dim < sol.nb_dims(); ++dim) {
            if (sol.space[dim] == 1) active_dims.push_back(dim);
        }

        float delta = 0.0f;
        const size_t n_rows = sol.csv.rows;

        // We only need to check the distance from the moving node to every other node
        #pragma omp parallel for reduction(+:delta) schedule(static)
        for (size_t i = 0; i < n_rows; ++i) {
            if (i == (size_t)node_idx) continue; // Skip self

            int group_other = m[i];

            // If the other node is not in the old or new group, the edge status (crossing vs internal) doesn't change.
            if (group_other != old_group && group_other != new_group) continue;

            float dist = 0.0f;
            #pragma omp simd reduction(+:dist)
            for (size_t d_i = 0; d_i < active_dims.size(); ++d_i) {
                int dim = active_dims[d_i];
                dist += std::fabsf(sol.csv.data[node_idx * sol.csv.cols + dim] - sol.csv.data[i * sol.csv.cols + dim]);
            }

            if (group_other == old_group) {
                delta += dist; //  If other node is in old_group: Edge was Internal (0), now becomes Crossing (+dist)
            } else {
                delta -= dist; //If other node is in new_group: Edge was Crossing (dist), now becomes Internal (-dist)
            }
        }
        return delta;
    }

};

#endif //COMPARING_NODES_BACKEND_CUT_HPP