#ifndef RT_RENDER_CPU_BACKEND_HPP
#define RT_RENDER_CPU_BACKEND_HPP

#include <memory>

#include "../../component/hittable_list.hpp"
#include "../render_backend.hpp"

class camera;

class CpuBackend final : public RenderBackend {
public:
    CpuBackend();
    ~CpuBackend() override;

    std::string_view name() const noexcept override { return "cpu"; }
    void prepare(
        const SceneData &scene,
        const CameraSettings &camera_settings,
        const RenderSettings &settings) override;
    RenderFrame render(int frame_index, int frame_count) override;

private:
    hittable_list world_;
    std::unique_ptr<camera> camera_;
    CameraSettings base_camera_settings_;
};

#endif
