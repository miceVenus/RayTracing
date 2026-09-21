#ifndef RT_CORE_SCENE_HPP
#define RT_CORE_SCENE_HPP

#include <cstdint>
#include <type_traits>
#include <vector>

struct Vec3Data {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

using ColorData = Vec3Data;

enum class MaterialKind : std::uint32_t {
    lambertian = 0,
    metal = 1,
    dielectric = 2,
};

struct MaterialData {
    MaterialKind kind = MaterialKind::lambertian;
    ColorData albedo{0.8, 0.3, 0.3};
    double fuzz = 0.0;
    double eta = 1.5;
};

struct SphereData {
    Vec3Data center{0.0, 0.0, -1.0};
    double radius = 0.5;
    MaterialData material;
};

struct SceneData {
    std::vector<SphereData> spheres;
};

struct CameraSettings {
    double aspect_ratio = 16.0 / 9.0;
    int image_width = 400;
    double vfov = 20.0;
    Vec3Data lookfrom{13.0, 2.0, 3.0};
    Vec3Data lookat{0.0, 0.0, 0.0};
    Vec3Data vup{0.0, 1.0, 0.0};
    double defocus_angle = 0.6;
    double focus_dist = 10.0;

    int image_height() const {
        const int height = static_cast<int>(image_width / aspect_ratio);
        return height < 1 ? 1 : height;
    }
};

struct RenderSettings {
    int samples_per_pixel = 500;
    int max_depth = 50;
    int thread_count = 16;
    std::uint64_t seed = 1;
};

static_assert(std::is_trivially_copyable<Vec3Data>::value, "Vec3Data must be safe to copy to a device buffer");
static_assert(std::is_trivially_copyable<SphereData>::value, "SphereData must be safe to copy to a device buffer");
static_assert(std::is_trivially_copyable<MaterialData>::value, "MaterialData must be safe to copy to a device buffer");

#endif
