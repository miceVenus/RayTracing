#ifndef COLOR_HPP
#define COLOR_HPP

#include "vec3.hpp"

using color = vec3;

inline double linear_to_gamma(double linear){
    if(linear > 0)
        return std::sqrt(linear);

    return 0;
}

#endif
