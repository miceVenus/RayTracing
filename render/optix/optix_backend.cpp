#include "optix_backend.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../camera_path.hpp"
#include "../../component/vec3.hpp"
#include "optix_shared.hpp"

namespace {

void check_cuda(cudaError_t result, const char *operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

void check_optix(OptixResult result, const char *operation) {
    if (result != OPTIX_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with OptiX error " +
            std::to_string(static_cast<int>(result)));
    }
}

OptixVec3 to_optix_vec(const Vec3Data &value) {
    return {static_cast<float>(value.x),
            static_cast<float>(value.y),
            static_cast<float>(value.z)};
}

OptixVec3 to_optix_vec(const vec3 &value) {
    return {static_cast<float>(value.x()),
            static_cast<float>(value.y()),
            static_cast<float>(value.z())};
}

OptixVec3 subtract(OptixVec3 a, OptixVec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

OptixVec3 add(OptixVec3 a, OptixVec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

OptixVec3 multiply(OptixVec3 value, float factor) {
    return {value.x * factor, value.y * factor, value.z * factor};
}

float dot(OptixVec3 a, OptixVec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

OptixVec3 normalize(OptixVec3 value) {
    const float length = std::sqrt(dot(value, value));
    if (length <= 0.0f) {
        throw std::runtime_error("OptiX camera basis contains a zero-length vector");
    }
    return multiply(value, 1.0f / length);
}

std::string read_ptx() {
    std::ifstream input(RT_OPTIX_PTX_PATH, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open OptiX device program: " RT_OPTIX_PTX_PATH);
    }
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

template <typename T>
void allocate_device(T **pointer, std::size_t count, const char *operation) {
    check_cuda(cudaMalloc(reinterpret_cast<void **>(pointer), count * sizeof(T)), operation);
}

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) EmptySbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

} // namespace

struct OptixBackend::Impl {
    OptixDeviceContext context = nullptr;
    OptixModule module = nullptr;
    OptixProgramGroup raygen_group = nullptr;
    OptixProgramGroup miss_group = nullptr;
    OptixProgramGroup hit_group = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixShaderBindingTable sbt{};
    OptixTraversableHandle traversable = 0;

    OptixSphereData *device_spheres = nullptr;
    OptixAabb *device_aabbs = nullptr;
    void *accel_buffer = nullptr;
    OptixLaunchParams *device_params = nullptr;
    unsigned char *device_image = nullptr;
    EmptySbtRecord *device_raygen_record = nullptr;
    EmptySbtRecord *device_miss_record = nullptr;
    EmptySbtRecord *device_hit_record = nullptr;
    cudaEvent_t event_start = nullptr;
    cudaEvent_t event_stop = nullptr;

    CameraSettings base_camera;
    RenderSettings render_settings;
    int image_width = 0;
    int image_height = 0;
    std::vector<OptixSphereData> host_spheres;

    ~Impl() {
        cleanup();
    }

    void cleanup() noexcept {
        if (event_stop != nullptr) cudaEventDestroy(event_stop);
        if (event_start != nullptr) cudaEventDestroy(event_start);
        if (device_hit_record != nullptr) cudaFree(device_hit_record);
        if (device_miss_record != nullptr) cudaFree(device_miss_record);
        if (device_raygen_record != nullptr) cudaFree(device_raygen_record);
        if (device_image != nullptr) cudaFree(device_image);
        if (device_params != nullptr) cudaFree(device_params);
        if (accel_buffer != nullptr) cudaFree(accel_buffer);
        if (device_aabbs != nullptr) cudaFree(device_aabbs);
        if (device_spheres != nullptr) cudaFree(device_spheres);
        if (pipeline != nullptr) optixPipelineDestroy(pipeline);
        if (hit_group != nullptr) optixProgramGroupDestroy(hit_group);
        if (miss_group != nullptr) optixProgramGroupDestroy(miss_group);
        if (raygen_group != nullptr) optixProgramGroupDestroy(raygen_group);
        if (module != nullptr) optixModuleDestroy(module);
        if (context != nullptr) optixDeviceContextDestroy(context);

        event_stop = nullptr;
        event_start = nullptr;
        device_hit_record = nullptr;
        device_miss_record = nullptr;
        device_raygen_record = nullptr;
        device_image = nullptr;
        device_params = nullptr;
        accel_buffer = nullptr;
        device_aabbs = nullptr;
        device_spheres = nullptr;
        pipeline = nullptr;
        hit_group = nullptr;
        miss_group = nullptr;
        raygen_group = nullptr;
        module = nullptr;
        context = nullptr;
    }
};

OptixBackend::OptixBackend() = default;
OptixBackend::~OptixBackend() = default;

void OptixBackend::prepare(
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
            "OptiX backend requires an available NVIDIA GPU and CUDA driver");
    }
    check_cuda(cudaSetDevice(0), "select CUDA device 0");
    check_cuda(cudaFree(nullptr), "initialize CUDA context");
    check_optix(optixInit(), "initialize OptiX");

    CUcontext cuda_context = nullptr;
    if (cuCtxGetCurrent(&cuda_context) != CUDA_SUCCESS || cuda_context == nullptr) {
        throw std::runtime_error("could not get the current CUDA context for OptiX");
    }
    OptixDeviceContextOptions context_options{};
    check_optix(optixDeviceContextCreate(cuda_context, &context_options, &next->context),
                "create OptiX device context");

    next->host_spheres.reserve(scene.spheres.size());
    std::vector<OptixAabb> host_aabbs;
    host_aabbs.reserve(scene.spheres.size());
    for (const SphereData &sphere : scene.spheres) {
        const float radius = static_cast<float>(sphere.radius);
        if (radius <= 0.0f) {
            continue;
        }
        OptixSphereData data{};
        data.center = to_optix_vec(sphere.center);
        data.radius = radius;
        data.material_kind = static_cast<std::uint32_t>(sphere.material.kind);
        data.albedo = to_optix_vec(sphere.material.albedo);
        data.fuzz = static_cast<float>(sphere.material.fuzz);
        data.eta = static_cast<float>(sphere.material.eta);
        next->host_spheres.push_back(data);

        const OptixVec3 extent{radius, radius, radius};
        const OptixVec3 lower = subtract(data.center, extent);
        const OptixVec3 upper = add(data.center, extent);
        host_aabbs.push_back({lower.x, lower.y, lower.z, upper.x, upper.y, upper.z});
    }
    if (next->host_spheres.empty()) {
        throw std::runtime_error("OptiX backend requires at least one sphere");
    }

    allocate_device(&next->device_spheres, next->host_spheres.size(), "allocate OptiX spheres");
    allocate_device(&next->device_aabbs, host_aabbs.size(), "allocate OptiX sphere bounds");
    check_cuda(cudaMemcpy(next->device_spheres, next->host_spheres.data(),
                          next->host_spheres.size() * sizeof(OptixSphereData),
                          cudaMemcpyHostToDevice), "copy OptiX spheres");
    check_cuda(cudaMemcpy(next->device_aabbs, host_aabbs.data(),
                          host_aabbs.size() * sizeof(OptixAabb),
                          cudaMemcpyHostToDevice), "copy OptiX sphere bounds");

    OptixBuildInput build_input{};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    CUdeviceptr aabb_buffer = reinterpret_cast<CUdeviceptr>(next->device_aabbs);
    unsigned int geometry_flags[] = {OPTIX_GEOMETRY_FLAG_NONE};
    build_input.customPrimitiveArray.aabbBuffers = &aabb_buffer;
    build_input.customPrimitiveArray.numPrimitives =
        static_cast<unsigned int>(next->host_spheres.size());
    build_input.customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
    build_input.customPrimitiveArray.flags = geometry_flags;
    build_input.customPrimitiveArray.numSbtRecords = 1;
    build_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
    build_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = 0;
    build_input.customPrimitiveArray.sbtIndexOffsetStrideInBytes = 0;

    OptixAccelBuildOptions accel_options{};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes accel_sizes{};
    check_optix(optixAccelComputeMemoryUsage(
        next->context, &accel_options, &build_input, 1, &accel_sizes),
        "compute OptiX acceleration structure memory usage");
    void *scratch_raw = nullptr;
    check_cuda(cudaMalloc(&scratch_raw, accel_sizes.tempSizeInBytes),
               "allocate OptiX acceleration structure scratch buffer");
    std::unique_ptr<void, cudaError_t (*)(void *)> scratch(scratch_raw, &cudaFree);
    check_cuda(cudaMalloc(&next->accel_buffer, accel_sizes.outputSizeInBytes),
               "allocate OptiX acceleration structure");
    const OptixResult build_result = optixAccelBuild(
        next->context,
        0,
        &accel_options,
        &build_input,
        1,
        reinterpret_cast<CUdeviceptr>(scratch.get()),
        accel_sizes.tempSizeInBytes,
        reinterpret_cast<CUdeviceptr>(next->accel_buffer),
        accel_sizes.outputSizeInBytes,
        &next->traversable,
        nullptr,
        0);
    check_cuda(cudaStreamSynchronize(0), "wait for OptiX acceleration structure build");
    check_cuda(cudaFree(scratch.release()), "free OptiX acceleration structure scratch buffer");
    check_optix(build_result, "build OptiX acceleration structure");

    const std::string ptx = read_ptx();
    OptixModuleCompileOptions module_options{};
    module_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeline_options{};
    pipeline_options.usesMotionBlur = 0;
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.numPayloadValues = 2;
    pipeline_options.numAttributeValues = 2;
    pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    char log[4096]{};
    std::size_t log_size = sizeof(log);
    check_optix(optixModuleCreate(
        next->context,
        &module_options,
        &pipeline_options,
        ptx.data(),
        ptx.size(),
        log,
        &log_size,
        &next->module),
        log_size > 1 ? log : "create OptiX module");

    OptixProgramGroupOptions group_options{};
    OptixProgramGroupDesc program_descriptions[3]{};
    program_descriptions[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    program_descriptions[0].raygen.module = next->module;
    program_descriptions[0].raygen.entryFunctionName = "__raygen__render";
    program_descriptions[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    program_descriptions[1].miss.module = next->module;
    program_descriptions[1].miss.entryFunctionName = "__miss__background";
    program_descriptions[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    program_descriptions[2].hitgroup.moduleCH = next->module;
    program_descriptions[2].hitgroup.entryFunctionNameCH = "__closesthit__sphere";
    program_descriptions[2].hitgroup.moduleIS = next->module;
    program_descriptions[2].hitgroup.entryFunctionNameIS = "__intersection__sphere";
    OptixProgramGroup groups[] = {
        next->raygen_group,
        next->miss_group,
        next->hit_group};
    log_size = sizeof(log);
    check_optix(optixProgramGroupCreate(
        next->context,
        program_descriptions,
        3,
        &group_options,
        log,
        &log_size,
        groups),
        log_size > 1 ? log : "create OptiX program groups");
    next->raygen_group = groups[0];
    next->miss_group = groups[1];
    next->hit_group = groups[2];

    OptixPipelineLinkOptions link_options{};
    link_options.maxTraceDepth = 1;
    log_size = sizeof(log);
    check_optix(optixPipelineCreate(
        next->context,
        &pipeline_options,
        &link_options,
        groups,
        3,
        log,
        &log_size,
        &next->pipeline),
        log_size > 1 ? log : "create OptiX pipeline");
    check_optix(optixPipelineSetStackSize(next->pipeline, 0, 0, 2048, 1),
                "set OptiX pipeline stack size");

    EmptySbtRecord host_raygen{};
    EmptySbtRecord host_miss{};
    EmptySbtRecord host_hit{};
    check_optix(optixSbtRecordPackHeader(next->raygen_group, &host_raygen),
                "pack OptiX raygen SBT record");
    check_optix(optixSbtRecordPackHeader(next->miss_group, &host_miss),
                "pack OptiX miss SBT record");
    check_optix(optixSbtRecordPackHeader(next->hit_group, &host_hit),
                "pack OptiX hitgroup SBT record");
    allocate_device(&next->device_raygen_record, 1, "allocate OptiX raygen SBT record");
    allocate_device(&next->device_miss_record, 1, "allocate OptiX miss SBT record");
    allocate_device(&next->device_hit_record, 1, "allocate OptiX hit SBT record");
    check_cuda(cudaMemcpy(next->device_raygen_record, &host_raygen, sizeof(host_raygen),
                          cudaMemcpyHostToDevice), "copy OptiX raygen SBT record");
    check_cuda(cudaMemcpy(next->device_miss_record, &host_miss, sizeof(host_miss),
                          cudaMemcpyHostToDevice), "copy OptiX miss SBT record");
    check_cuda(cudaMemcpy(next->device_hit_record, &host_hit, sizeof(host_hit),
                          cudaMemcpyHostToDevice), "copy OptiX hit SBT record");
    next->sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(next->device_raygen_record);
    next->sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(next->device_miss_record);
    next->sbt.missRecordStrideInBytes = sizeof(EmptySbtRecord);
    next->sbt.missRecordCount = 1;
    next->sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(next->device_hit_record);
    next->sbt.hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
    next->sbt.hitgroupRecordCount = 1;

    allocate_device(&next->device_params, 1, "allocate OptiX launch parameters");
    allocate_device(&next->device_image,
        static_cast<std::size_t>(next->image_width) * next->image_height * 4,
        "allocate OptiX output image");
    check_cuda(cudaEventCreate(&next->event_start), "create OptiX start event");
    check_cuda(cudaEventCreate(&next->event_stop), "create OptiX stop event");
    impl_ = std::move(next);
}

RenderFrame OptixBackend::render(int frame_index, int frame_count) {
    if (!impl_ || impl_->pipeline == nullptr) {
        throw std::logic_error("OptixBackend::prepare must be called before render");
    }

    const auto wall_start = std::chrono::steady_clock::now();
    const CameraSettings camera = orbit_camera_for_frame(
        impl_->base_camera, frame_index, frame_count);
    const int width = camera.image_width;
    const int height = camera.image_height();
    RenderFrame frame;
    frame.image.width = width;
    frame.image.height = height;
    frame.image.rgba.resize(static_cast<std::size_t>(width) * height * 4);
    const vec3 lookfrom(camera.lookfrom.x, camera.lookfrom.y, camera.lookfrom.z);
    const vec3 lookat(camera.lookat.x, camera.lookat.y, camera.lookat.z);
    const vec3 vup(camera.vup.x, camera.vup.y, camera.vup.z);
    const vec3 w = unit(lookfrom - lookat);
    const vec3 u = unit(cross(vup, w));
    const vec3 v = cross(w, u);

    const double theta = deg2rad(camera.vfov);
    const double viewport_height = 2.0 * std::tan(theta / 2.0) * camera.focus_dist;
    const double viewport_width = viewport_height *
        (static_cast<double>(width) / static_cast<double>(height));
    const vec3 viewport_u = u * viewport_width;
    const vec3 viewport_v = v * -viewport_height;
    const vec3 pixel_delta_u = viewport_u / width;
    const vec3 pixel_delta_v = viewport_v / height;
    const vec3 viewport_left_up = lookfrom - w * camera.focus_dist -
        viewport_u / 2.0 - viewport_v / 2.0;
    const vec3 pixel00 = viewport_left_up + (pixel_delta_u + pixel_delta_v) * 0.5;
    const double defocus_radius = camera.focus_dist * std::tan(deg2rad(camera.defocus_angle / 2.0));
    const vec3 defocus_disk_u = u * defocus_radius;
    const vec3 defocus_disk_v = v * defocus_radius;

    OptixLaunchParams params{};
    params.spheres = impl_->device_spheres;
    params.rgba = impl_->device_image;
    params.traversable = static_cast<std::uint64_t>(impl_->traversable);
    params.width = static_cast<unsigned int>(width);
    params.height = static_cast<unsigned int>(height);
    params.samples_per_pixel = static_cast<unsigned int>(impl_->render_settings.samples_per_pixel);
    params.max_depth = static_cast<unsigned int>(impl_->render_settings.max_depth);
    params.seed = impl_->render_settings.seed;
    params.camera_center = to_optix_vec(lookfrom);
    params.pixel00 = to_optix_vec(pixel00);
    params.pixel_delta_u = to_optix_vec(pixel_delta_u);
    params.pixel_delta_v = to_optix_vec(pixel_delta_v);
    params.defocus_disk_u = to_optix_vec(defocus_disk_u);
    params.defocus_disk_v = to_optix_vec(defocus_disk_v);
    params.sphere_count = static_cast<unsigned int>(impl_->host_spheres.size());
    check_cuda(cudaMemcpy(impl_->device_params, &params, sizeof(params), cudaMemcpyHostToDevice),
               "copy OptiX launch parameters");

    check_cuda(cudaEventRecord(impl_->event_start), "record OptiX start event");
    check_optix(optixLaunch(
        impl_->pipeline,
        0,
        reinterpret_cast<CUdeviceptr>(impl_->device_params),
        sizeof(params),
        &impl_->sbt,
        static_cast<unsigned int>(width),
        static_cast<unsigned int>(height),
        1),
        "launch OptiX renderer");
    check_cuda(cudaEventRecord(impl_->event_stop), "record OptiX stop event");
    check_cuda(cudaEventSynchronize(impl_->event_stop), "wait for OptiX render");
    float backend_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&backend_ms, impl_->event_start, impl_->event_stop),
               "measure OptiX render time");
    check_cuda(cudaMemcpy(frame.image.rgba.data(), impl_->device_image,
                          frame.image.rgba.size(), cudaMemcpyDeviceToHost),
               "copy OptiX image to host");

    const auto wall_end = std::chrono::steady_clock::now();
    frame.timings = {
        static_cast<double>(backend_ms),
        std::chrono::duration<double, std::milli>(wall_end - wall_start).count()};
    return frame;
}
