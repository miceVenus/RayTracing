#ifndef RT_RENDER_CUDA_SHARED_HPP
#define RT_RENDER_CUDA_SHARED_HPP

#include <cstdint>

struct CudaVec3 {
    float x;
    float y;
    float z;
};

struct CudaSphereData {
    CudaVec3 center;
    float radius;
    std::uint32_t material_kind;
    CudaVec3 albedo;
    float fuzz;
    float eta;
};

struct CudaBvhNode {
    CudaVec3 bounds_min;
    CudaVec3 bounds_max;
    int left;
    int first;
    int count;
    int escape;
};

struct CudaLaunchParams {
    const CudaSphereData *spheres;
    const CudaBvhNode *nodes;
    unsigned char *rgba;
    unsigned int width;
    unsigned int height;
    unsigned int sphere_count;
    unsigned int node_count;
    unsigned int samples_per_pixel;
    unsigned int max_depth;
    std::uint64_t seed;
    CudaVec3 camera_center;
    CudaVec3 pixel00;
    CudaVec3 pixel_delta_u;
    CudaVec3 pixel_delta_v;
    CudaVec3 defocus_disk_u;
    CudaVec3 defocus_disk_v;
};

#endif
