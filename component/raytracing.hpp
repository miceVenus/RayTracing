#ifndef RAYTRACING_HPP
#define RAYTRACING_HPP

#include <random>
#include <cmath>
#include <ostream>
#include <memory>
#include <cmath>
#include <limits>


#include <ctime>



#include <cstdint>

struct pcg {
    public:
        pcg(uint64_t seed, uint64_t seq = 1) {
            state = 0;
            inc = (seq << 1u) | 1u;
            next_u32();
            state += seed;
            next_u32();
        }

        uint32_t next_u32() {
            uint64_t oldstate = state;
            state = oldstate * 6364136223846793005ULL + inc;

            uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
            uint32_t rot = static_cast<uint32_t>(oldstate >> 59u);

            return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
        }

        float next_float() {
            return (next_u32() >> 8) * (1.0f / 16777216.0f);
        }

        double next_double() {
            return (next_u32() >> 11) * (1.0 / 2097152.0);
        }

    private:
        uint64_t state;
        uint64_t inc;
};

const double infinity   = std::numeric_limits<double>::infinity();
const double pi         = 3.1415926535897932385;

inline double deg2rad(double degree){
    return degree * pi / 180;
}


// inline double random_double(){
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<double> distrib(0.0, 1.0);

//     return distrib(gen);
// }

inline pcg *&active_random_generator() {
    static thread_local pcg *active_rng = nullptr;
    return active_rng;
}

inline double random_double(){
    if (pcg *active_rng = active_random_generator()) {
        return active_rng->next_double();
    }
    static thread_local pcg fallback_rng(
        (uint64_t(std::random_device{}()) << 32) ^ uint64_t(std::random_device{}()));
    return fallback_rng.next_double();
}

class pcg_scope {
public:
    explicit pcg_scope(pcg &rng) : previous_(active_random_generator()) {
        active_random_generator() = &rng;
    }

    pcg_scope(const pcg_scope &) = delete;
    pcg_scope &operator=(const pcg_scope &) = delete;

    ~pcg_scope() {
        active_random_generator() = previous_;
    }

private:
    pcg *previous_;
};

// inline double random_double(pcg &gen){
//     return gen.next_double();
// }

inline double random_double(double min, double max){
    return min + (max - min) * random_double();
}

// inline double random_double(double min, double max, pcg &rand){
//     return min + (max - min) * random_double(rand);
// }


#endif
