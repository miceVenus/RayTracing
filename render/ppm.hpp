#ifndef RT_RENDER_PPM_HPP
#define RT_RENDER_PPM_HPP

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>

#include "image.hpp"

inline void write_ppm(const std::string &path, const RenderImage &image) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("could not open image output: " + path);
    }

    output << "P3\n" << image.width << ' ' << image.height << "\n255\n";
    for (std::size_t pixel = 0; pixel < static_cast<std::size_t>(image.width) * image.height; ++pixel) {
        const std::size_t offset = pixel * 4;
        output << static_cast<unsigned int>(image.rgba[offset]) << ' '
               << static_cast<unsigned int>(image.rgba[offset + 1]) << ' '
               << static_cast<unsigned int>(image.rgba[offset + 2]) << '\n';
    }
    if (!output) {
        throw std::runtime_error("failed while writing image output: " + path);
    }
}

#endif
