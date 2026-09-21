#ifndef RT_RENDER_OPTIX_SHARED_HPP
#define RT_RENDER_OPTIX_SHARED_HPP

#include <cstdint>

struct OptixVec3 {
    float x;
    float y;
    float z;
};

struct OptixSphereData {
    OptixVec3 center;
    float radius;
    std::uint32_t material_kind;
    OptixVec3 albedo;
    float fuzz;
    float eta;
};

struct OptixLaunchParams {
    OptixSphereData *spheres;
    unsigned char *rgba;
    std::uint64_t traversable;
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_pixel;
    unsigned int max_depth;
    std::uint64_t seed;
    OptixVec3 camera_center;
    OptixVec3 pixel00;
    OptixVec3 pixel_delta_u;
    OptixVec3 pixel_delta_v;
    OptixVec3 defocus_disk_u;
    OptixVec3 defocus_disk_v;
    unsigned int sphere_count;
};

#endif
