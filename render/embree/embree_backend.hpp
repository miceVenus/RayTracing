#ifndef RT_RENDER_EMBREE_BACKEND_HPP
#define RT_RENDER_EMBREE_BACKEND_HPP

#include <memory>

#include "../render_backend.hpp"

class EmbreeBackend final : public RenderBackend {
public:
    EmbreeBackend();
    ~EmbreeBackend() override;

    std::string_view name() const noexcept override { return "embree"; }
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
