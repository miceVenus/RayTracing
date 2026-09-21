#include "embree_backend.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if RT_EMBREE_VERSION_MAJOR >= 4
#include <embree4/rtcore.h>
#else
#include <embree3/rtcore.h>
#endif

#include "../../component/camera.hpp"
#include "../../component/hittable.hpp"
#include "../../component/material.hpp"
#include "../camera_path.hpp"

namespace {

vec3 to_vec3(const Vec3Data &value) {
    return vec3(value.x, value.y, value.z);
}

color to_color(const ColorData &value) {
    return color(value.x, value.y, value.z);
}

std::shared_ptr<material> make_material(const MaterialData &data) {
    switch (data.kind) {
        case MaterialKind::metal:
            return std::make_shared<metal>(to_color(data.albedo), data.fuzz);
        case MaterialKind::dielectric:
            return std::make_shared<dielectric>(data.eta);
        case MaterialKind::lambertian:
        default:
            return std::make_shared<lambertian>(to_color(data.albedo));
    }
}

class EmbreeWorld final : public hittable {
public:
    explicit EmbreeWorld(const SceneData &scene) {
        device_ = rtcNewDevice(nullptr);
        if (device_ == nullptr) {
            throw std::runtime_error("Embree could not create a device");
        }

        scene_ = rtcNewScene(device_);
        if (scene_ == nullptr) {
            rtcReleaseDevice(device_);
            device_ = nullptr;
            throw std::runtime_error("Embree could not create a scene");
        }

        RTCGeometry geometry = nullptr;
        try {
            rtcSetSceneBuildQuality(scene_, RTC_BUILD_QUALITY_HIGH);
            std::vector<const SphereData *> active_spheres;
            active_spheres.reserve(scene.spheres.size());
            for (const SphereData &sphere : scene.spheres) {
                if (sphere.radius > 0.0) {
                    active_spheres.push_back(&sphere);
                }
            }

            if (!active_spheres.empty()) {
                geometry = rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_SPHERE_POINT);
                if (geometry == nullptr) {
                    throw std::runtime_error("Embree could not create sphere geometry");
                }

                constexpr std::size_t vertex_stride = 4 * sizeof(float);
                auto *vertices = static_cast<float *>(rtcSetNewGeometryBuffer(
                    geometry,
                    RTC_BUFFER_TYPE_VERTEX,
                    0,
                    RTC_FORMAT_FLOAT4,
                    vertex_stride,
                    active_spheres.size()));
                if (vertices == nullptr) {
                    throw std::runtime_error("Embree could not allocate sphere vertices");
                }

                materials_.reserve(active_spheres.size());
                for (std::size_t index = 0; index < active_spheres.size(); ++index) {
                    const SphereData &sphere = *active_spheres[index];
                    vertices[index * 4 + 0] = static_cast<float>(sphere.center.x);
                    vertices[index * 4 + 1] = static_cast<float>(sphere.center.y);
                    vertices[index * 4 + 2] = static_cast<float>(sphere.center.z);
                    vertices[index * 4 + 3] = static_cast<float>(sphere.radius);
                    materials_.push_back(make_material(sphere.material));
                }

                rtcCommitGeometry(geometry);
                sphere_geometry_id_ = rtcAttachGeometry(scene_, geometry);
                rtcReleaseGeometry(geometry);
                geometry = nullptr;
            }

            rtcCommitScene(scene_);
            const RTCError error = rtcGetDeviceError(device_);
            if (error != RTC_ERROR_NONE) {
                throw std::runtime_error("Embree failed while building the scene (error " +
                    std::to_string(static_cast<int>(error)) + ")");
            }
        } catch (...) {
            if (geometry != nullptr) {
                rtcReleaseGeometry(geometry);
            }
            rtcReleaseScene(scene_);
            rtcReleaseDevice(device_);
            scene_ = nullptr;
            device_ = nullptr;
            throw;
        }
    }

    ~EmbreeWorld() override {
        if (scene_ != nullptr) {
            rtcReleaseScene(scene_);
        }
        if (device_ != nullptr) {
            rtcReleaseDevice(device_);
        }
    }

    bool hit(const ray &input, const interval &ray_t, hit_record &record) const override {
        RTCRayHit rayhit{};
        rayhit.ray.org_x = static_cast<float>(input.origin().x());
        rayhit.ray.org_y = static_cast<float>(input.origin().y());
        rayhit.ray.org_z = static_cast<float>(input.origin().z());
        rayhit.ray.dir_x = static_cast<float>(input.direction().x());
        rayhit.ray.dir_y = static_cast<float>(input.direction().y());
        rayhit.ray.dir_z = static_cast<float>(input.direction().z());
        rayhit.ray.tnear = static_cast<float>(ray_t.min);
        rayhit.ray.tfar = static_cast<float>(ray_t.max);
        rayhit.ray.time = 0.0f;
        rayhit.ray.mask = UINT32_MAX;
        rayhit.ray.id = 0;
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
        for (unsigned int &instance_id : rayhit.hit.instID) {
            instance_id = RTC_INVALID_GEOMETRY_ID;
        }

#if RT_EMBREE_VERSION_MAJOR >= 4
        RTCIntersectArguments arguments;
        rtcInitIntersectArguments(&arguments);
        rtcIntersect1(scene_, &rayhit, &arguments);
#else
        RTCIntersectContext context;
        rtcInitIntersectContext(&context);
        rtcIntersect1(scene_, &rayhit, &context);
#endif

        if (rayhit.hit.geomID != sphere_geometry_id_ ||
            rayhit.hit.primID >= materials_.size()) {
            return false;
        }

        const double distance = rayhit.ray.tfar;
        record.t = distance;
        record.p = input.at(distance);
        const vec3 outward_normal(
            rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z);
        record.set_face_normal(input, unit(outward_normal));
        record.mat = materials_[rayhit.hit.primID];
        return true;
    }

private:
    RTCDevice device_ = nullptr;
    RTCScene scene_ = nullptr;
    unsigned int sphere_geometry_id_ = RTC_INVALID_GEOMETRY_ID;
    std::vector<std::shared_ptr<material>> materials_;
};

void apply_camera(camera &target, const CameraSettings &settings) {
    target.aspect_ratio = settings.aspect_ratio;
    target.image_width = settings.image_width;
    target.vfov = settings.vfov;
    target.lookfrom = to_vec3(settings.lookfrom);
    target.lookat = to_vec3(settings.lookat);
    target.vup = to_vec3(settings.vup);
    target.defocus_angle = settings.defocus_angle;
    target.focus_dist = settings.focus_dist;
}

} // namespace

struct EmbreeBackend::Impl {
    std::unique_ptr<EmbreeWorld> world;
    std::unique_ptr<camera> camera_instance;
    CameraSettings base_camera;
};

EmbreeBackend::EmbreeBackend() = default;
EmbreeBackend::~EmbreeBackend() = default;

void EmbreeBackend::prepare(
    const SceneData &scene,
    const CameraSettings &camera_settings,
    const RenderSettings &settings) {
    auto next = std::make_unique<Impl>();
    next->world = std::make_unique<EmbreeWorld>(scene);
    next->camera_instance = std::make_unique<camera>(settings.thread_count, settings.seed);
    next->base_camera = camera_settings;
    apply_camera(*next->camera_instance, camera_settings);
    next->camera_instance->samples_per_pixel = settings.samples_per_pixel;
    next->camera_instance->max_depth = settings.max_depth;
    impl_ = std::move(next);
}

RenderFrame EmbreeBackend::render(int frame_index, int frame_count) {
    if (!impl_ || !impl_->camera_instance || !impl_->world) {
        throw std::logic_error("EmbreeBackend::prepare must be called before render");
    }

    const CameraSettings frame_camera = orbit_camera_for_frame(
        impl_->base_camera, frame_index, frame_count);
    apply_camera(*impl_->camera_instance, frame_camera);

    const auto start = std::chrono::steady_clock::now();
    RenderImage image = impl_->camera_instance->render_to_image(*impl_->world);
    const auto end = std::chrono::steady_clock::now();
    const double wall_ms = std::chrono::duration<double, std::milli>(end - start).count();

    RenderFrame frame;
    frame.image = std::move(image);
    frame.timings = {wall_ms, wall_ms};
    return frame;
}
