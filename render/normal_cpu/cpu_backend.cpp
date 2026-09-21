#include "cpu_backend.hpp"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <utility>

#include "../../component/camera.hpp"
#include "../../component/material.hpp"
#include "../../component/shpere.hpp"
#include "../camera_path.hpp"

namespace {

vec3 to_vec3(const Vec3Data &value) {
    return vec3(value.x, value.y, value.z);
}

color to_color(const ColorData &value) {
    return color(value.x, value.y, value.z);
}

} // namespace

CpuBackend::CpuBackend() = default;
CpuBackend::~CpuBackend() = default;

void CpuBackend::prepare(
    const SceneData &scene,
    const CameraSettings &camera_settings,
    const RenderSettings &settings) {
    base_camera_settings_ = camera_settings;
    world_.clear();
    for (const SphereData &sphere_data : scene.spheres) {
        std::shared_ptr<material> material_instance;
        switch (sphere_data.material.kind) {
            case MaterialKind::metal:
                material_instance = std::make_shared<metal>(
                    to_color(sphere_data.material.albedo), sphere_data.material.fuzz);
                break;
            case MaterialKind::dielectric:
                material_instance = std::make_shared<dielectric>(sphere_data.material.eta);
                break;
            case MaterialKind::lambertian:
            default:
                material_instance = std::make_shared<lambertian>(to_color(sphere_data.material.albedo));
                break;
        }

        world_.add(std::make_shared<sphere>(
            to_vec3(sphere_data.center), sphere_data.radius, std::move(material_instance)));
    }

    camera_ = std::make_unique<camera>(settings.thread_count, settings.seed);
    camera_->aspect_ratio = camera_settings.aspect_ratio;
    camera_->image_width = camera_settings.image_width;
    camera_->samples_per_pixel = settings.samples_per_pixel;
    camera_->max_depth = settings.max_depth;
    camera_->vfov = camera_settings.vfov;
    camera_->lookfrom = to_vec3(camera_settings.lookfrom);
    camera_->lookat = to_vec3(camera_settings.lookat);
    camera_->vup = to_vec3(camera_settings.vup);
    camera_->defocus_angle = camera_settings.defocus_angle;
    camera_->focus_dist = camera_settings.focus_dist;
}

RenderFrame CpuBackend::render(int frame_index, int frame_count) {
    if (!camera_) {
        throw std::logic_error("CpuBackend::prepare must be called before render");
    }

    const CameraSettings frame_camera = orbit_camera_for_frame(
        base_camera_settings_, frame_index, frame_count);
    camera_->lookfrom = to_vec3(frame_camera.lookfrom);
    camera_->lookat = to_vec3(frame_camera.lookat);

    const auto start = std::chrono::steady_clock::now();
    RenderImage image = camera_->render_to_image(world_);
    const auto end = std::chrono::steady_clock::now();
    const double wall_ms = std::chrono::duration<double, std::milli>(end - start).count();

    RenderFrame frame;
    frame.image = std::move(image);
    frame.timings = {wall_ms, wall_ms};
    return frame;
}
