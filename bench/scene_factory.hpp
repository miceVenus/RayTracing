#ifndef RT_BENCH_SCENE_FACTORY_HPP
#define RT_BENCH_SCENE_FACTORY_HPP

#include <cmath>
#include <cstdint>

#include "../core/scene.hpp"
#include "../component/raytracing.hpp"

namespace benchmark_scene {

inline double random(pcg &rng, double min_value = 0.0, double max_value = 1.0) {
    return min_value + (max_value - min_value) * rng.next_double();
}

inline ColorData random_color(pcg &rng, double min_value = 0.0, double max_value = 1.0) {
    return {
        random(rng, min_value, max_value),
        random(rng, min_value, max_value),
        random(rng, min_value, max_value)
    };
}

inline ColorData multiply(const ColorData &a, const ColorData &b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

inline SceneData make_reference_scene(std::uint64_t seed) {
    SceneData scene;
    scene.spheres.reserve(500);
    pcg rng(seed);

    MaterialData ground;
    ground.kind = MaterialKind::lambertian;
    ground.albedo = {0.5, 0.5, 0.5};
    scene.spheres.push_back({{0.0, -1000.0, 0.0}, 1000.0, ground});

    for (int a = -11; a < 11; ++a) {
        for (int b = -11; b < 11; ++b) {
            const double choose_mat = random(rng);
            const Vec3Data center{
                a + 0.9 * random(rng),
                0.2,
                b + 0.9 * random(rng)
            };
            const double dx = center.x - 4.0;
            const double dy = center.y - 0.2;
            const double dz = center.z;
            if (std::sqrt(dx * dx + dy * dy + dz * dz) <= 0.9) {
                continue;
            }

            MaterialData material;
            if (choose_mat < 0.6) {
                material.kind = MaterialKind::lambertian;
                const ColorData first_color = random_color(rng);
                const ColorData second_color = random_color(rng);
                material.albedo = multiply(first_color, second_color);
            } else if (choose_mat < 0.8) {
                material.kind = MaterialKind::metal;
                material.albedo = random_color(rng, 0.5, 1.0);
                material.fuzz = random(rng, 0.0, 0.5);
            } else {
                material.kind = MaterialKind::dielectric;
                material.eta = 1.5;
            }

            scene.spheres.push_back({center, 0.2, material});
        }
    }

    MaterialData glass;
    glass.kind = MaterialKind::dielectric;
    glass.eta = 1.5;
    scene.spheres.push_back({{0.0, 1.0, 0.0}, 1.0, glass});

    MaterialData diffuse;
    diffuse.kind = MaterialKind::lambertian;
    diffuse.albedo = {0.4, 0.2, 0.1};
    scene.spheres.push_back({{-4.0, 1.0, 0.0}, 1.0, diffuse});

    MaterialData metal;
    metal.kind = MaterialKind::metal;
    metal.albedo = {0.7, 0.6, 0.5};
    metal.fuzz = 0.0;
    scene.spheres.push_back({{4.0, 1.0, 0.0}, 1.0, metal});

    return scene;
}

} // namespace benchmark_scene

#endif
