#include <iostream>

#include "argparse.hpp"
#include "csv_loader.hpp"
#include "Solution.hpp"
#include "Cut.hpp"
#include "rng_freezer.hpp"


std::vector<Solution> neighborhood(Solution& s) {
    std::vector<Solution> neighborhood;

    for (int i = 0; i < s.nb_dims(); ++i) {
        if (s.space[i] == 0) {
            neighborhood.emplace_back(s, i , 1);
            neighborhood.emplace_back(s, i , -1);
        }
        if (s.space[i] == 1 and s.nb_dims_comp() > 1) {
            neighborhood.emplace_back(s, i , 0);
            neighborhood.emplace_back(s, i , -1);
        }
        if (s.space[i] == -1 and s.nb_dims_clust() > 1) {
            neighborhood.emplace_back(s, i , 0);
            neighborhood.emplace_back(s, i , 1);
        }
    }

    return neighborhood;
}

int main(int argc, char *argv[]) {
    argparse::ArgumentParser program("comparing-nodes");

    program.add_argument("--dataset")
        .default_value(std::string{"/Users/chanson/CLionProjects/comparing_nodes_backend/test_data/iris.csv"})   // might otherwise be type const char* leading to an error when trying program.get<std::string>
        .help("specify the dataset file (csv)");

    program.add_argument("--delimiter")
    .default_value(',')
    .action([](const std::string& value) { return value.at(0); }) //only supports single char delimiters
    .help("specify the dataset file delimiter (single char only)");

    program.add_argument("--k")
    .default_value(int{3})   // might otherwise be type const char* leading to an error when trying program.get<std::string>
    .scan<'d', int>()
    .help("specify the number of clusters");

    program.add_argument("--ss-iters")
    .default_value(int{25})
    .scan<'d', int>()
    .help("specify the number of iterations for subspace search");

    program.add_argument("--cut-alpha")
    .default_value(float{0.98})
    .scan<'g', float>()
    .help("exp cooling for min cut");

    program.add_argument("--cut-iter")
    .default_value(float{10})
    .scan<'g', float>()
    .help("exp cooling for min cut");

    program.add_argument("--cut-temp")
    .default_value(float{1})
    .scan<'g', float>()
    .help("exp cooling for min cut");

    program.add_argument("--freeze-rng")
    .default_value(false)
    .implicit_value(true)
    .help("freeze rng to a default seed");

    program.add_argument("--seed")
    .default_value(int{42})
    .scan<'d', int>()
    .help("specify the seed for the rng");

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    try {
        const auto path = program.get<std::string>("--dataset");
        const char delim = program.get<char>("--delimiter");
        const int k = program.get<int>("--k");
        const int max_iters = program.get<int>("--ss-iters");
        const float alpha = program.get<float>("--cut-alpha");
        const float iter_factor = program.get<float>("--cut-iter");
        const float temp_factor = program.get<float>("--cut-temp");
        rng_freezer::freeze_rng = program.get<bool>("--freeze-rng");
        if (rng_freezer::freeze_rng) {
            rng_freezer::set_seed(program.get<int>("--seed"));
        }

        csv::CSVNumericData data = csv::load_numeric_csv(path, delim);

        // Print matrix shape
        std::cout << "Dataset Shape ("
                  << data.rows << " rows x "
                  << data.cols << " cols)\n";
        // Print numeric headers
        std::cout << "Numeric columns loaded: ";
        for (const auto& h : data.headers) {
            std::cout << " " << h << ",";
        }
        std::cout << std::endl;

        Solution start(data.cols, data); // random solution

        Solution current = start;
        current.membership = Cut::min_cut(current, k, alpha, iter_factor, temp_factor);
        current.nb_clusters = k;
        float current_value = current.obj();

        current.debug_print();

        for (int iter = 0; iter < max_iters; ++iter) {
            std::cout << "LS iter " << iter <<std::endl;
            // build neighbors
            std::vector<Solution> neigh = neighborhood(current);
            if (neigh.empty()) {
                std::cout << "No neighbors found, stopping." << std::endl;
                break;
            }
            //evaluate them
            float best_value = 0;
            int best_sol_idx = 0;
            for (int i = 0; i < neigh.size(); ++i) {
                neigh[i].membership = Cut::min_cut(neigh[i], k, alpha, iter_factor, temp_factor);
                neigh[i].nb_clusters = k;
                //neigh[i].debug_print();
                float obj_val = neigh[i].obj();
                if (obj_val > best_value) {
                    best_value = obj_val;
                    best_sol_idx = i;
                    if (best_value > current_value) {
                        break;
                    }
                }
            }

            if (best_value <= current_value) {
                std::cout << "stopped early " << iter << std::endl;
                current = neigh[best_sol_idx];
                break;
            }
            current = neigh[best_sol_idx];
            current_value = best_value;
        }

        current.debug_print();

        std::cout << "[SOLUTION] " << current << std::endl;
        std::cout << "[OBJ] " << current.obj() << std::endl;
        std::cout << "[CLUSTERS] ";
        std::cout << "[";
        for (std::size_t i = 0; i < current.membership.size(); ++i) {
            if (i > 0) std::cout << ",";
            std::cout << current.membership[i];
        }
        std::cout << "]";


    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
    }
}
