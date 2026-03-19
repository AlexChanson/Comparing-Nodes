#ifndef RNG_FREEZER_HPP
#define RNG_FREEZER_HPP

#include <random>

namespace rng_freezer {
    inline bool freeze_rng = false;
    inline std::mt19937 frozen_engine;

    inline void set_seed(unsigned int seed) {
        frozen_engine.seed(seed);
    }
}

#endif //RNG_FREEZER_HPP
