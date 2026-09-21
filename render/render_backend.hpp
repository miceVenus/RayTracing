#ifndef RT_RENDER_BACKEND_HPP
#define RT_RENDER_BACKEND_HPP

#include <string_view>

#include "../core/scene.hpp"
#include "image.hpp"

class RenderBackend {
public:
    virtual ~RenderBackend() = default;

    virtual std::string_view name() const noexcept = 0;
    virtual void prepare(
        const SceneData &scene,
        const CameraSettings &camera,
        const RenderSettings &settings) = 0;
    // frame_count defines one deterministic camera orbit; index is in [0, frame_count).
    virtual RenderFrame render(int frame_index, int frame_count) = 0;
};

#endif
