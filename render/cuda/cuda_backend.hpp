#ifndef RT_RENDER_CUDA_BACKEND_HPP
#define RT_RENDER_CUDA_BACKEND_HPP

#include <memory>

#include "../render_backend.hpp"

class CudaBackend final : public RenderBackend {
public:
    CudaBackend();
    ~CudaBackend() override;

    std::string_view name() const noexcept override { return "cuda"; }
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
