#ifndef RT_RENDER_CAMERA_PATH_HPP
#define RT_RENDER_CAMERA_PATH_HPP

#include <cmath>

#include "../core/scene.hpp"

inline CameraSettings orbit_camera_for_frame(
    const CameraSettings &base,
    int frame_index,
    int frame_count) {
    CameraSettings camera = base;
    if (frame_count <= 0) {
        return camera;
    }

    constexpr double two_pi = 6.28318530717958647692;
    const double angle = two_pi * static_cast<double>(frame_index) /
        static_cast<double>(frame_count);
    const double x = base.lookfrom.x - base.lookat.x;
    const double z = base.lookfrom.z - base.lookat.z;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);

    camera.lookfrom.x = base.lookat.x + cosine * x - sine * z;
    camera.lookfrom.y = base.lookfrom.y;
    camera.lookfrom.z = base.lookat.z + sine * x + cosine * z;
    return camera;
}

#endif
