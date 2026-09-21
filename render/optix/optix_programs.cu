#include <optix.h>
#include <optix_device.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "optix_shared.hpp"

extern "C" __constant__ OptixLaunchParams params;

namespace {

constexpr unsigned int invalid_index = 0xffffffffu;
constexpr float ray_min = 0.001f;

__device__ __forceinline__ OptixVec3 add(OptixVec3 a, OptixVec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ __forceinline__ OptixVec3 subtract(OptixVec3 a, OptixVec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ __forceinline__ OptixVec3 multiply(OptixVec3 a, float s) {
    return {a.x * s, a.y * s, a.z * s};
}

__device__ __forceinline__ OptixVec3 multiply(OptixVec3 a, OptixVec3 b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ __forceinline__ float dot(OptixVec3 a, OptixVec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float length_squared(OptixVec3 value) {
    return dot(value, value);
}

__device__ __forceinline__ OptixVec3 normalize(OptixVec3 value) {
    return multiply(value, rsqrtf(length_squared(value)));
}

__device__ __forceinline__ OptixVec3 cross(OptixVec3 a, OptixVec3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

struct Pcg32 {
    std::uint64_t state;
    std::uint64_t increment;

    __device__ explicit Pcg32(std::uint64_t seed) : state(0), increment(3) {
        next_u32();
        state += seed;
        next_u32();
    }

    __device__ std::uint32_t next_u32() {
        const std::uint64_t old_state = state;
        state = old_state * 6364136223846793005ULL + increment;
        const std::uint32_t shifted = static_cast<std::uint32_t>(
            ((old_state >> 18u) ^ old_state) >> 27u);
        const std::uint32_t rotation = static_cast<std::uint32_t>(old_state >> 59u);
        return (shifted >> rotation) | (shifted << ((-rotation) & 31));
    }

    __device__ float next_float() {
        return static_cast<float>(next_u32() >> 11) * (1.0f / 2097152.0f);
    }
};

__device__ std::uint64_t pixel_seed(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__device__ OptixVec3 random_vec3(Pcg32 &rng, float min_value, float max_value) {
    const float scale = max_value - min_value;
    return {min_value + scale * rng.next_float(),
            min_value + scale * rng.next_float(),
            min_value + scale * rng.next_float()};
}

__device__ OptixVec3 random_in_unit_sphere(Pcg32 &rng) {
    for (;;) {
        const OptixVec3 point = random_vec3(rng, -1.0f, 1.0f);
        const float squared_length = length_squared(point);
        if (squared_length > 0.0f && squared_length < 1.0f) {
            return point;
        }
    }
}

__device__ OptixVec3 random_unit_vector(Pcg32 &rng) {
    return normalize(random_in_unit_sphere(rng));
}

__device__ OptixVec3 random_in_unit_disk(Pcg32 &rng) {
    for (;;) {
        const OptixVec3 point{
            -1.0f + 2.0f * rng.next_float(),
            -1.0f + 2.0f * rng.next_float(),
            0.0f};
        if (length_squared(point) < 1.0f) {
            return point;
        }
    }
}

__device__ OptixVec3 reflect(OptixVec3 value, OptixVec3 normal) {
    return subtract(value, multiply(normal, 2.0f * dot(value, normal)));
}

__device__ OptixVec3 refract(OptixVec3 unit_direction, OptixVec3 normal, float ratio) {
    const float cosine = fminf(-dot(unit_direction, normal), 1.0f);
    const OptixVec3 perpendicular = multiply(
        add(unit_direction, multiply(normal, cosine)), ratio);
    const float parallel_scale = -sqrtf(fabsf(1.0f - length_squared(perpendicular)));
    return add(perpendicular, multiply(normal, parallel_scale));
}

__device__ float reflectance(float cosine, float ratio) {
    float r0 = (1.0f - ratio) / (1.0f + ratio);
    r0 *= r0;
    const float complement = 1.0f - cosine;
    return r0 + (1.0f - r0) * complement * complement * complement * complement * complement;
}

__device__ OptixVec3 background(OptixVec3 direction) {
    const OptixVec3 unit_direction = normalize(direction);
    const float amount = 0.5f * (unit_direction.y + 1.0f);
    return add(multiply({1.0f, 1.0f, 1.0f}, 1.0f - amount),
               multiply({0.5f, 0.7f, 1.0f}, amount));
}

__device__ void store_color(unsigned char *output, OptixVec3 color) {
    const float r = sqrtf(fmaxf(color.x, 0.0f));
    const float g = sqrtf(fmaxf(color.y, 0.0f));
    const float b = sqrtf(fmaxf(color.z, 0.0f));
    output[0] = static_cast<unsigned char>(256.0f * fminf(r, 0.999f));
    output[1] = static_cast<unsigned char>(256.0f * fminf(g, 0.999f));
    output[2] = static_cast<unsigned char>(256.0f * fminf(b, 0.999f));
    output[3] = 255;
}

} // namespace

extern "C" __global__ void __raygen__render() {
    const uint3 launch = optixGetLaunchIndex();
    if (launch.x >= params.width || launch.y >= params.height) {
        return;
    }

    const unsigned int pixel = launch.y * params.width + launch.x;
    Pcg32 rng(pixel_seed(params.seed ^ static_cast<std::uint64_t>(pixel)));
    OptixVec3 pixel_color{0.0f, 0.0f, 0.0f};

    for (unsigned int sample = 0; sample < params.samples_per_pixel; ++sample) {
        const float offset_x = rng.next_float() - 0.5f;
        const float offset_y = rng.next_float() - 0.5f;
        const OptixVec3 pixel_sample = add(
            params.pixel00,
            add(multiply(params.pixel_delta_u, static_cast<float>(launch.x) + offset_x),
                multiply(params.pixel_delta_v, static_cast<float>(launch.y) + offset_y)));

        OptixVec3 ray_origin = params.camera_center;
        if (length_squared(params.defocus_disk_u) > 0.0f ||
            length_squared(params.defocus_disk_v) > 0.0f) {
            const OptixVec3 disk = random_in_unit_disk(rng);
            ray_origin = add(ray_origin,
                add(multiply(params.defocus_disk_u, disk.x),
                    multiply(params.defocus_disk_v, disk.y)));
        }
        OptixVec3 ray_direction = subtract(pixel_sample, ray_origin);
        OptixVec3 throughput{1.0f, 1.0f, 1.0f};
        OptixVec3 sample_color{0.0f, 0.0f, 0.0f};

        for (unsigned int depth = 0; depth < params.max_depth; ++depth) {
            unsigned int hit_index = invalid_index;
            unsigned int hit_t_bits = 0;
            optixTrace(
                static_cast<OptixTraversableHandle>(params.traversable),
                make_float3(ray_origin.x, ray_origin.y, ray_origin.z),
                make_float3(ray_direction.x, ray_direction.y, ray_direction.z),
                ray_min,
                1.0e30f,
                0.0f,
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE,
                0,
                1,
                0,
                hit_index,
                hit_t_bits);

            if (hit_index == invalid_index || hit_index >= params.sphere_count) {
                sample_color = multiply(throughput, background(ray_direction));
                break;
            }

            const OptixSphereData sphere = params.spheres[hit_index];
            const float hit_t = __uint_as_float(hit_t_bits);
            const OptixVec3 hit_point = add(ray_origin, multiply(ray_direction, hit_t));
            const OptixVec3 outward_normal = normalize(
                multiply(subtract(hit_point, sphere.center), 1.0f / sphere.radius));
            const bool front_face = dot(ray_direction, outward_normal) < 0.0f;
            const OptixVec3 normal = front_face ? outward_normal : multiply(outward_normal, -1.0f);

            if (sphere.material_kind == 0u) {
                throughput = multiply(throughput, sphere.albedo);
                OptixVec3 scattered = add(normal, random_unit_vector(rng));
                if (fabsf(scattered.x) < 1.0e-8f &&
                    fabsf(scattered.y) < 1.0e-8f &&
                    fabsf(scattered.z) < 1.0e-8f) {
                    scattered = normal;
                }
                ray_origin = hit_point;
                ray_direction = scattered;
            } else if (sphere.material_kind == 1u) {
                throughput = multiply(throughput, sphere.albedo);
                const OptixVec3 reflected = normalize(reflect(normalize(ray_direction), normal));
                ray_direction = add(reflected,
                    multiply(random_unit_vector(rng), fminf(sphere.fuzz, 1.0f)));
                if (dot(ray_direction, normal) <= 0.0f) {
                    break;
                }
                ray_origin = hit_point;
            } else {
                const float ratio = front_face ? 1.0f / sphere.eta : sphere.eta;
                const OptixVec3 unit_direction = normalize(ray_direction);
                const float cosine = fminf(-dot(unit_direction, normal), 1.0f);
                const float sine = sqrtf(1.0f - cosine * cosine);
                const bool cannot_refract = ratio * sine > 1.0f;
                ray_direction = (cannot_refract || reflectance(cosine, ratio) > rng.next_float())
                    ? reflect(unit_direction, normal)
                    : refract(unit_direction, normal, ratio);
                ray_origin = hit_point;
            }
        }

        pixel_color = add(pixel_color, sample_color);
    }

    const float scale = 1.0f / static_cast<float>(params.samples_per_pixel);
    store_color(params.rgba + static_cast<std::size_t>(pixel) * 4,
                multiply(pixel_color, scale));
}

extern "C" __global__ void __miss__background() {
    optixSetPayload_0(invalid_index);
}

extern "C" __global__ void __closesthit__sphere() {
    optixSetPayload_0(optixGetPrimitiveIndex());
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
}

extern "C" __global__ void __intersection__sphere() {
    const unsigned int primitive = optixGetPrimitiveIndex();
    if (primitive >= params.sphere_count) {
        return;
    }

    const OptixSphereData sphere = params.spheres[primitive];
    const float3 object_origin = optixGetObjectRayOrigin();
    const float3 object_direction = optixGetObjectRayDirection();
    const OptixVec3 origin{object_origin.x, object_origin.y, object_origin.z};
    const OptixVec3 direction{object_direction.x, object_direction.y, object_direction.z};
    const OptixVec3 from_center = subtract(origin, sphere.center);
    const float a = length_squared(direction);
    const float half_b = dot(from_center, direction);
    const float c = length_squared(from_center) - sphere.radius * sphere.radius;
    const float discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0f) {
        return;
    }

    const float root_delta = sqrtf(discriminant);
    float root = (-half_b - root_delta) / a;
    if (root < optixGetRayTmin() || root > optixGetRayTmax()) {
        root = (-half_b + root_delta) / a;
        if (root < optixGetRayTmin() || root > optixGetRayTmax()) {
            return;
        }
    }
    optixReportIntersection(root, 0);
}
