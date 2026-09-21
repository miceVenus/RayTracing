#ifndef RT_RENDER_OPTIX_BACKEND_HPP
#define RT_RENDER_OPTIX_BACKEND_HPP

#include <memory>

#include "../render_backend.hpp"

class OptixBackend final : public RenderBackend {
public:
    OptixBackend();
    ~OptixBackend() override;

    std::string_view name() const noexcept override { return "optix"; }
    void prepare(
        const SceneData &scene,
        const CameraSettings &camera,
        const RenderSettings &settings) override;
    RenderFrame render(int frame_index, int frame_count) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif
