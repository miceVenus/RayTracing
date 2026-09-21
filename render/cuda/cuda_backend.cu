#include "cuda_backend.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../camera_path.hpp"
#include "cuda_shared.hpp"

namespace {

constexpr float ray_min = 0.001f;
constexpr float ray_max = 1.0e30f;
constexpr int bvh_leaf_size = 4;
constexpr unsigned int threads_per_block = 256;

void check_cuda(cudaError_t result, const char *operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

template <typename T>
void allocate_device(T **pointer, std::size_t count, const char *operation) {
    if (count == 0) {
        throw std::invalid_argument(std::string(operation) + " cannot allocate an empty buffer");
    }
    check_cuda(cudaMalloc(reinterpret_cast<void **>(pointer), count * sizeof(T)), operation);
}

float component(const CudaVec3 &value, int axis) {
    if (axis == 0) return value.x;
    if (axis == 1) return value.y;
    return value.z;
}

CudaVec3 add(CudaVec3 a, CudaVec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

CudaVec3 subtract(CudaVec3 a, CudaVec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

struct HostVec3 {
    double x;
    double y;
    double z;
};

HostVec3 from_scene_vec(const Vec3Data &value) {
    return {value.x, value.y, value.z};
}

HostVec3 host_add(HostVec3 a, HostVec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

HostVec3 host_subtract(HostVec3 a, HostVec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

HostVec3 host_multiply(HostVec3 value, double factor) {
    return {value.x * factor, value.y * factor, value.z * factor};
}

HostVec3 host_cross(HostVec3 a, HostVec3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

HostVec3 host_normalize(HostVec3 value) {
    const double length = std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
    if (length <= 0.0) {
        throw std::runtime_error("CUDA camera basis contains a zero-length vector");
    }
    return host_multiply(value, 1.0 / length);
}

CudaVec3 to_cuda_vec(HostVec3 value) {
    return {static_cast<float>(value.x),
            static_cast<float>(value.y),
            static_cast<float>(value.z)};
}

CudaBvhNode empty_node() {
    return {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, -1, 0, 0, -1};
}

int build_bvh_node(
    int begin,
    int end,
    const std::vector<CudaSphereData> &spheres,
    std::vector<unsigned int> &indices,
    std::vector<CudaBvhNode> &nodes) {
    const int node_index = static_cast<int>(nodes.size());
    nodes.push_back(empty_node());

    CudaVec3 bounds_min{
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity()};
    CudaVec3 bounds_max{
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()};
    CudaVec3 centroid_min = bounds_min;
    CudaVec3 centroid_max = bounds_max;

    for (int position = begin; position < end; ++position) {
        const CudaSphereData &sphere = spheres[indices[static_cast<std::size_t>(position)]];
        const CudaVec3 radius{sphere.radius, sphere.radius, sphere.radius};
        const CudaVec3 lower = subtract(sphere.center, radius);
        const CudaVec3 upper = add(sphere.center, radius);
        bounds_min = {std::min(bounds_min.x, lower.x),
                      std::min(bounds_min.y, lower.y),
                      std::min(bounds_min.z, lower.z)};
        bounds_max = {std::max(bounds_max.x, upper.x),
                      std::max(bounds_max.y, upper.y),
                      std::max(bounds_max.z, upper.z)};
        centroid_min = {std::min(centroid_min.x, sphere.center.x),
                        std::min(centroid_min.y, sphere.center.y),
                        std::min(centroid_min.z, sphere.center.z)};
        centroid_max = {std::max(centroid_max.x, sphere.center.x),
                        std::max(centroid_max.y, sphere.center.y),
                        std::max(centroid_max.z, sphere.center.z)};
    }

    const int count = end - begin;
    CudaBvhNode node{bounds_min, bounds_max, -1, begin, count, -1};
    if (count > bvh_leaf_size) {
        const CudaVec3 centroid_extent = subtract(centroid_max, centroid_min);
        int split_axis = 0;
        if (centroid_extent.y > centroid_extent.x) split_axis = 1;
        if (component(centroid_extent, 2) > component(centroid_extent, split_axis)) {
            split_axis = 2;
        }

        const int middle = begin + count / 2;
        auto first = indices.begin() + begin;
        auto mid = indices.begin() + middle;
        auto last = indices.begin() + end;
        std::nth_element(first, mid, last,
            [&spheres, split_axis](unsigned int lhs, unsigned int rhs) {
                const float lhs_value = component(spheres[lhs].center, split_axis);
                const float rhs_value = component(spheres[rhs].center, split_axis);
                return lhs_value == rhs_value ? lhs < rhs : lhs_value < rhs_value;
            });

        node.left = build_bvh_node(begin, middle, spheres, indices, nodes);
        build_bvh_node(middle, end, spheres, indices, nodes);
        node.first = 0;
        node.count = 0;
    }

    // Preorder layout makes the first node after this subtree a threaded escape
    // link. A left subtree escapes directly to its parent's right subtree.
    node.escape = static_cast<int>(nodes.size());
    nodes[static_cast<std::size_t>(node_index)] = node;
    return node_index;
}

struct CameraLaunchData {
    CudaVec3 center;
    CudaVec3 pixel00;
    CudaVec3 pixel_delta_u;
    CudaVec3 pixel_delta_v;
    CudaVec3 defocus_disk_u;
    CudaVec3 defocus_disk_v;
};

CameraLaunchData make_camera_launch_data(const CameraSettings &camera) {
    constexpr double pi = 3.14159265358979323846;
    const HostVec3 lookfrom = from_scene_vec(camera.lookfrom);
    const HostVec3 lookat = from_scene_vec(camera.lookat);
    const HostVec3 vup = from_scene_vec(camera.vup);
    const HostVec3 w = host_normalize(host_subtract(lookfrom, lookat));
    const HostVec3 u = host_normalize(host_cross(vup, w));
    const HostVec3 v = host_cross(w, u);

    const double theta = camera.vfov * pi / 180.0;
    const double viewport_height =
        2.0 * std::tan(theta / 2.0) * camera.focus_dist;
    const double viewport_width = viewport_height *
        (static_cast<double>(camera.image_width) /
         static_cast<double>(camera.image_height()));
    const HostVec3 viewport_u = host_multiply(u, viewport_width);
    const HostVec3 viewport_v = host_multiply(v, -viewport_height);
    const HostVec3 pixel_delta_u =
        host_multiply(viewport_u, 1.0 / static_cast<double>(camera.image_width));
    const HostVec3 pixel_delta_v =
        host_multiply(viewport_v, 1.0 / static_cast<double>(camera.image_height()));
    const HostVec3 viewport_left_up = host_subtract(
        host_subtract(
            host_subtract(lookfrom, host_multiply(w, camera.focus_dist)),
            host_multiply(viewport_u, 0.5)),
        host_multiply(viewport_v, 0.5));
    const HostVec3 pixel00 = host_add(
        viewport_left_up,
        host_multiply(host_add(pixel_delta_u, pixel_delta_v), 0.5));
    const double defocus_radius = camera.focus_dist *
        std::tan((camera.defocus_angle / 2.0) * pi / 180.0);

    return {
        to_cuda_vec(lookfrom),
        to_cuda_vec(pixel00),
        to_cuda_vec(pixel_delta_u),
        to_cuda_vec(pixel_delta_v),
        to_cuda_vec(host_multiply(u, defocus_radius)),
        to_cuda_vec(host_multiply(v, defocus_radius))};
}

__device__ __forceinline__ CudaVec3 device_add(CudaVec3 a, CudaVec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ __forceinline__ CudaVec3 device_subtract(CudaVec3 a, CudaVec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ __forceinline__ CudaVec3 device_multiply(CudaVec3 value, float factor) {
    return {value.x * factor, value.y * factor, value.z * factor};
}

__device__ __forceinline__ CudaVec3 device_multiply(CudaVec3 a, CudaVec3 b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ __forceinline__ float device_dot(CudaVec3 a, CudaVec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float device_length_squared(CudaVec3 value) {
    return device_dot(value, value);
}

__device__ __forceinline__ CudaVec3 device_normalize(CudaVec3 value) {
    const float length_squared = device_length_squared(value);
    if (length_squared <= 0.0f) return {0.0f, 0.0f, 0.0f};
    return device_multiply(value, rsqrtf(length_squared));
}

struct DevicePcg32 {
    std::uint64_t state;
    std::uint64_t increment;

    __device__ explicit DevicePcg32(std::uint64_t seed) : state(0), increment(3) {
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
        return static_cast<float>(next_u32() >> 11u) * (1.0f / 2097152.0f);
    }
};

__device__ std::uint64_t pixel_seed(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

__device__ CudaVec3 random_vec3(DevicePcg32 &rng, float min_value, float max_value) {
    const float scale = max_value - min_value;
    return {min_value + scale * rng.next_float(),
            min_value + scale * rng.next_float(),
            min_value + scale * rng.next_float()};
}

__device__ CudaVec3 random_in_unit_sphere(DevicePcg32 &rng) {
    for (;;) {
        const CudaVec3 point = random_vec3(rng, -1.0f, 1.0f);
        const float squared_length = device_length_squared(point);
        if (squared_length > 0.0f && squared_length < 1.0f) return point;
    }
}

__device__ CudaVec3 random_unit_vector(DevicePcg32 &rng) {
    return device_normalize(random_in_unit_sphere(rng));
}

__device__ CudaVec3 random_in_unit_disk(DevicePcg32 &rng) {
    for (;;) {
        const CudaVec3 point{-1.0f + 2.0f * rng.next_float(),
                             -1.0f + 2.0f * rng.next_float(),
                             0.0f};
        if (device_length_squared(point) < 1.0f) return point;
    }
}

__device__ CudaVec3 reflect(CudaVec3 value, CudaVec3 normal) {
    return device_subtract(value, device_multiply(normal, 2.0f * device_dot(value, normal)));
}

__device__ CudaVec3 refract(CudaVec3 unit_direction, CudaVec3 normal, float ratio) {
    const float cosine = fminf(-device_dot(unit_direction, normal), 1.0f);
    const CudaVec3 perpendicular = device_multiply(
        device_add(unit_direction, device_multiply(normal, cosine)), ratio);
    const float parallel_scale = -sqrtf(fabsf(1.0f - device_length_squared(perpendicular)));
    return device_add(perpendicular, device_multiply(normal, parallel_scale));
}

__device__ float reflectance(float cosine, float ratio) {
    float r0 = (1.0f - ratio) / (1.0f + ratio);
    r0 *= r0;
    const float complement = 1.0f - cosine;
    return r0 + (1.0f - r0) * complement * complement * complement * complement * complement;
}

__device__ CudaVec3 background(CudaVec3 direction) {
    const CudaVec3 unit_direction = device_normalize(direction);
    const float amount = 0.5f * (unit_direction.y + 1.0f);
    return device_add(device_multiply({1.0f, 1.0f, 1.0f}, 1.0f - amount),
                      device_multiply({0.5f, 0.7f, 1.0f}, amount));
}

__device__ bool intersect_aabb(
    const CudaBvhNode &node,
    CudaVec3 origin,
    CudaVec3 direction,
    CudaVec3 inverse_direction,
    float ray_tmin,
    float ray_tmax) {
    float near_t = ray_tmin;
    float far_t = ray_tmax;
    for (int axis = 0; axis < 3; ++axis) {
        const float origin_component = axis == 0 ? origin.x : (axis == 1 ? origin.y : origin.z);
        const float direction_component = axis == 0 ? direction.x :
            (axis == 1 ? direction.y : direction.z);
        const float lower = axis == 0 ? node.bounds_min.x :
            (axis == 1 ? node.bounds_min.y : node.bounds_min.z);
        const float upper = axis == 0 ? node.bounds_max.x :
            (axis == 1 ? node.bounds_max.y : node.bounds_max.z);

        if (fabsf(direction_component) < 1.0e-20f) {
            if (origin_component < lower || origin_component > upper) return false;
            continue;
        }

        const float inverse_direction_component = axis == 0 ? inverse_direction.x :
            (axis == 1 ? inverse_direction.y : inverse_direction.z);
        float first = (lower - origin_component) * inverse_direction_component;
        float second = (upper - origin_component) * inverse_direction_component;
        if (first > second) {
            const float temporary = first;
            first = second;
            second = temporary;
        }
        near_t = fmaxf(near_t, first);
        far_t = fminf(far_t, second);
        if (far_t < near_t) return false;
    }
    return true;
}

__device__ bool intersect_sphere(
    const CudaSphereData &sphere,
    CudaVec3 origin,
    CudaVec3 direction,
    float ray_tmin,
    float ray_tmax,
    float &hit_t) {
    const CudaVec3 from_center = device_subtract(origin, sphere.center);
    const float a = device_length_squared(direction);
    const float half_b = device_dot(from_center, direction);
    const float c = device_length_squared(from_center) - sphere.radius * sphere.radius;
    const float discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0f || a <= 0.0f) return false;

    const float root_delta = sqrtf(discriminant);
    float root = (-half_b - root_delta) / a;
    if (root < ray_tmin || root > ray_tmax) {
        root = (-half_b + root_delta) / a;
        if (root < ray_tmin || root > ray_tmax) return false;
    }
    hit_t = root;
    return true;
}

__device__ bool trace_scene(
    const CudaLaunchParams &params,
    CudaVec3 origin,
    CudaVec3 direction,
    unsigned int &hit_sphere,
    float &hit_t) {
    int node_index = 0;
    float closest_t = ray_max;
    int closest_sphere = -1;
    const CudaVec3 inverse_direction{
        fabsf(direction.x) < 1.0e-20f ? 0.0f : 1.0f / direction.x,
        fabsf(direction.y) < 1.0e-20f ? 0.0f : 1.0f / direction.y,
        fabsf(direction.z) < 1.0e-20f ? 0.0f : 1.0f / direction.z};

    while (node_index >= 0 && static_cast<unsigned int>(node_index) < params.node_count) {
        const CudaBvhNode &node = params.nodes[node_index];
        if (!intersect_aabb(node, origin, direction, inverse_direction, ray_min, closest_t)) {
            node_index = node.escape;
            continue;
        }

        if (node.count > 0) {
            for (int offset = 0; offset < node.count; ++offset) {
                const int sphere_index = node.first + offset;
                float candidate_t = 0.0f;
                if (intersect_sphere(params.spheres[sphere_index], origin, direction,
                                     ray_min, closest_t, candidate_t)) {
                    closest_t = candidate_t;
                    closest_sphere = sphere_index;
                }
            }
            node_index = node.escape;
        } else {
            node_index = node.left;
        }
    }

    if (closest_sphere < 0) return false;
    hit_sphere = static_cast<unsigned int>(closest_sphere);
    hit_t = closest_t;
    return true;
}

__device__ void store_color(unsigned char *output, CudaVec3 color) {
    const float r = sqrtf(fmaxf(color.x, 0.0f));
    const float g = sqrtf(fmaxf(color.y, 0.0f));
    const float b = sqrtf(fmaxf(color.z, 0.0f));
    output[0] = static_cast<unsigned char>(256.0f * fminf(r, 0.999f));
    output[1] = static_cast<unsigned char>(256.0f * fminf(g, 0.999f));
    output[2] = static_cast<unsigned char>(256.0f * fminf(b, 0.999f));
    output[3] = 255;
}

__global__ void render_kernel(CudaLaunchParams params) {
    const unsigned int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int pixel_count = params.width * params.height;
    if (pixel >= pixel_count) return;

    const unsigned int x = pixel % params.width;
    const unsigned int y = pixel / params.width;
    DevicePcg32 rng(pixel_seed(params.seed ^ static_cast<std::uint64_t>(pixel)));
    CudaVec3 pixel_color{0.0f, 0.0f, 0.0f};

    for (unsigned int sample = 0; sample < params.samples_per_pixel; ++sample) {
        const float offset_x = rng.next_float() - 0.5f;
        const float offset_y = rng.next_float() - 0.5f;
        const CudaVec3 pixel_sample = device_add(
            params.pixel00,
            device_add(device_multiply(params.pixel_delta_u, static_cast<float>(x) + offset_x),
                       device_multiply(params.pixel_delta_v, static_cast<float>(y) + offset_y)));

        CudaVec3 ray_origin = params.camera_center;
        if (device_length_squared(params.defocus_disk_u) > 0.0f ||
            device_length_squared(params.defocus_disk_v) > 0.0f) {
            const CudaVec3 disk = random_in_unit_disk(rng);
            ray_origin = device_add(ray_origin,
                device_add(device_multiply(params.defocus_disk_u, disk.x),
                           device_multiply(params.defocus_disk_v, disk.y)));
        }
        CudaVec3 ray_direction = device_subtract(pixel_sample, ray_origin);
        CudaVec3 throughput{1.0f, 1.0f, 1.0f};
        CudaVec3 sample_color{0.0f, 0.0f, 0.0f};

        for (unsigned int depth = 0; depth < params.max_depth; ++depth) {
            unsigned int hit_index = 0;
            float hit_t = 0.0f;
            if (!trace_scene(params, ray_origin, ray_direction, hit_index, hit_t) ||
                hit_index >= params.sphere_count) {
                sample_color = device_multiply(throughput, background(ray_direction));
                break;
            }

            const CudaSphereData sphere = params.spheres[hit_index];
            const CudaVec3 hit_point = device_add(ray_origin, device_multiply(ray_direction, hit_t));
            const CudaVec3 outward_normal = device_normalize(
                device_multiply(device_subtract(hit_point, sphere.center), 1.0f / sphere.radius));
            const bool front_face = device_dot(ray_direction, outward_normal) < 0.0f;
            const CudaVec3 normal = front_face
                ? outward_normal
                : device_multiply(outward_normal, -1.0f);

            if (sphere.material_kind == 0u) {
                throughput = device_multiply(throughput, sphere.albedo);
                CudaVec3 scattered = device_add(normal, random_unit_vector(rng));
                if (fabsf(scattered.x) < 1.0e-8f &&
                    fabsf(scattered.y) < 1.0e-8f &&
                    fabsf(scattered.z) < 1.0e-8f) {
                    scattered = normal;
                }
                ray_origin = hit_point;
                ray_direction = scattered;
            } else if (sphere.material_kind == 1u) {
                throughput = device_multiply(throughput, sphere.albedo);
                const CudaVec3 reflected = device_normalize(
                    reflect(device_normalize(ray_direction), normal));
                ray_direction = device_add(reflected,
                    device_multiply(random_unit_vector(rng), fminf(sphere.fuzz, 1.0f)));
                if (device_dot(ray_direction, normal) <= 0.0f) break;
                ray_origin = hit_point;
            } else {
                const float ratio = front_face ? 1.0f / sphere.eta : sphere.eta;
                const CudaVec3 unit_direction = device_normalize(ray_direction);
                const float cosine = fminf(-device_dot(unit_direction, normal), 1.0f);
                const float sine = sqrtf(1.0f - cosine * cosine);
                const bool cannot_refract = ratio * sine > 1.0f;
                ray_direction = (cannot_refract || reflectance(cosine, ratio) > rng.next_float())
                    ? reflect(unit_direction, normal)
                    : refract(unit_direction, normal, ratio);
                ray_origin = hit_point;
            }
        }

        pixel_color = device_add(pixel_color, sample_color);
    }

    const float scale = 1.0f / static_cast<float>(params.samples_per_pixel);
    store_color(params.rgba + static_cast<std::size_t>(pixel) * 4,
                device_multiply(pixel_color, scale));
}

} // namespace

struct CudaBackend::Impl {
    CudaSphereData *device_spheres = nullptr;
    CudaBvhNode *device_nodes = nullptr;
    unsigned char *device_image = nullptr;
    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_stop = nullptr;
    CameraSettings base_camera;
    RenderSettings render_settings;
    int image_width = 0;
    int image_height = 0;
    unsigned int sphere_count = 0;
    unsigned int node_count = 0;

    ~Impl() {
        cleanup();
    }

    void cleanup() noexcept {
        if (event_stop != nullptr) cudaEventDestroy(event_stop);
        if (event_start != nullptr) cudaEventDestroy(event_start);
        if (device_image != nullptr) cudaFree(device_image);
        if (device_nodes != nullptr) cudaFree(device_nodes);
        if (device_spheres != nullptr) cudaFree(device_spheres);
        event_stop = nullptr;
        event_start = nullptr;
        device_image = nullptr;
        device_nodes = nullptr;
        device_spheres = nullptr;
    }
};

CudaBackend::CudaBackend() = default;
CudaBackend::~CudaBackend() = default;

void CudaBackend::prepare(
    const SceneData &scene,
    const CameraSettings &camera_settings,
    const RenderSettings &settings) {
    auto next = std::make_unique<Impl>();
    next->base_camera = camera_settings;
    next->render_settings = settings;
    next->image_width = camera_settings.image_width;
    next->image_height = camera_settings.image_height();

    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess || device_count < 1) {
        throw std::runtime_error(
            "CUDA backend requires an available NVIDIA GPU and CUDA driver");
    }
    check_cuda(cudaSetDevice(0), "select CUDA device 0");
    check_cuda(cudaFree(nullptr), "initialize CUDA context");

    std::vector<CudaSphereData> host_spheres;
    host_spheres.reserve(scene.spheres.size());
    for (const SphereData &source : scene.spheres) {
        const float radius = static_cast<float>(source.radius);
        if (radius <= 0.0f) continue;
        host_spheres.push_back({
            {static_cast<float>(source.center.x),
             static_cast<float>(source.center.y),
             static_cast<float>(source.center.z)},
            radius,
            static_cast<std::uint32_t>(source.material.kind),
            {static_cast<float>(source.material.albedo.x),
             static_cast<float>(source.material.albedo.y),
             static_cast<float>(source.material.albedo.z)},
            static_cast<float>(source.material.fuzz),
            static_cast<float>(source.material.eta)});
    }
    if (host_spheres.empty()) {
        throw std::runtime_error("CUDA backend requires at least one positive-radius sphere");
    }
    if (host_spheres.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("CUDA backend scene contains too many spheres for the BVH");
    }

    std::vector<unsigned int> indices(host_spheres.size());
    std::iota(indices.begin(), indices.end(), 0u);
    std::vector<CudaBvhNode> host_nodes;
    host_nodes.reserve(host_spheres.size() * 2);
    build_bvh_node(0, static_cast<int>(indices.size()), host_spheres, indices, host_nodes);

    std::vector<CudaSphereData> ordered_spheres;
    ordered_spheres.reserve(host_spheres.size());
    for (unsigned int index : indices) ordered_spheres.push_back(host_spheres[index]);

    if (host_nodes.size() > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
        throw std::runtime_error("CUDA backend BVH contains too many nodes");
    }
    next->sphere_count = static_cast<unsigned int>(ordered_spheres.size());
    next->node_count = static_cast<unsigned int>(host_nodes.size());

    allocate_device(&next->device_spheres, ordered_spheres.size(), "allocate CUDA spheres");
    check_cuda(cudaMemcpy(next->device_spheres, ordered_spheres.data(),
                          ordered_spheres.size() * sizeof(CudaSphereData),
                          cudaMemcpyHostToDevice), "copy CUDA spheres");
    allocate_device(&next->device_nodes, host_nodes.size(), "allocate CUDA BVH nodes");
    check_cuda(cudaMemcpy(next->device_nodes, host_nodes.data(),
                          host_nodes.size() * sizeof(CudaBvhNode),
                          cudaMemcpyHostToDevice), "copy CUDA BVH nodes");

    const std::size_t pixel_count = static_cast<std::size_t>(next->image_width) *
        static_cast<std::size_t>(next->image_height);
    if (pixel_count > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
        throw std::runtime_error("CUDA image has too many pixels for a one-dimensional launch");
    }
    allocate_device(&next->device_image, pixel_count * 4, "allocate CUDA image");
    check_cuda(cudaEventCreate(&next->event_start), "create CUDA start event");
    check_cuda(cudaEventCreate(&next->event_stop), "create CUDA stop event");
    impl_ = std::move(next);
}

RenderFrame CudaBackend::render(int frame_index, int frame_count) {
    if (!impl_ || impl_->device_image == nullptr) {
        throw std::logic_error("CudaBackend::prepare must be called before render");
    }

    const auto wall_start = std::chrono::steady_clock::now();
    const CameraSettings camera = orbit_camera_for_frame(
        impl_->base_camera, frame_index, frame_count);
    const CameraLaunchData camera_data = make_camera_launch_data(camera);

    RenderFrame frame;
    frame.image.width = impl_->image_width;
    frame.image.height = impl_->image_height;
    frame.image.rgba.resize(static_cast<std::size_t>(impl_->image_width) *
                            static_cast<std::size_t>(impl_->image_height) * 4);

    CudaLaunchParams params{};
    params.spheres = impl_->device_spheres;
    params.nodes = impl_->device_nodes;
    params.rgba = impl_->device_image;
    params.width = static_cast<unsigned int>(impl_->image_width);
    params.height = static_cast<unsigned int>(impl_->image_height);
    params.sphere_count = impl_->sphere_count;
    params.node_count = impl_->node_count;
    params.samples_per_pixel = static_cast<unsigned int>(impl_->render_settings.samples_per_pixel);
    params.max_depth = static_cast<unsigned int>(impl_->render_settings.max_depth);
    params.seed = impl_->render_settings.seed;
    params.camera_center = camera_data.center;
    params.pixel00 = camera_data.pixel00;
    params.pixel_delta_u = camera_data.pixel_delta_u;
    params.pixel_delta_v = camera_data.pixel_delta_v;
    params.defocus_disk_u = camera_data.defocus_disk_u;
    params.defocus_disk_v = camera_data.defocus_disk_v;

    const std::size_t pixel_count = static_cast<std::size_t>(impl_->image_width) *
        static_cast<std::size_t>(impl_->image_height);
    const unsigned int block_count = static_cast<unsigned int>(
        (pixel_count + threads_per_block - 1) / threads_per_block);

    check_cuda(cudaEventRecord(impl_->event_start), "record CUDA start event");
    render_kernel<<<block_count, threads_per_block>>>(params);
    check_cuda(cudaGetLastError(), "launch CUDA renderer");
    check_cuda(cudaEventRecord(impl_->event_stop), "record CUDA stop event");
    check_cuda(cudaEventSynchronize(impl_->event_stop), "wait for CUDA render");
    float backend_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&backend_ms, impl_->event_start, impl_->event_stop),
               "measure CUDA render time");
    check_cuda(cudaMemcpy(frame.image.rgba.data(), impl_->device_image,
                          frame.image.rgba.size(), cudaMemcpyDeviceToHost),
               "copy CUDA image to host");

    const auto wall_end = std::chrono::steady_clock::now();
    frame.timings = {
        static_cast<double>(backend_ms),
        std::chrono::duration<double, std::milli>(wall_end - wall_start).count()};
    return frame;
}
