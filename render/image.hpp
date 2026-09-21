#ifndef RT_RENDER_IMAGE_HPP
#define RT_RENDER_IMAGE_HPP

#include <cstdint>
#include <vector>

struct RenderImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgba;
};

struct RenderTimings {
    // Backend-reported execution time. A CUDA backend can fill this with event timing.
    double backend_ms = 0.0;
    // Host wall time spent inside the render call.
    double wall_ms = 0.0;
};

struct RenderFrame {
    RenderImage image;
    RenderTimings timings;
};

#endif
