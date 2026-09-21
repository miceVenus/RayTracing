#ifndef RT_BENCH_BENCHMARK_HPP
#define RT_BENCH_BENCHMARK_HPP

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../render/render_backend.hpp"

struct BenchmarkOptions {
    int warmup_iterations = 0;
    int iterations = 1;
    int frames = 1;
    int fps = 30;
};

struct BenchmarkReport {
    double prepare_ms = 0.0;
    std::vector<double> backend_samples_ms;
    std::vector<double> wall_samples_ms;
    RenderFrame last_frame;
    std::vector<RenderFrame> last_frames;
};

struct TimingSummary {
    double minimum_ms = 0.0;
    double median_ms = 0.0;
    double mean_ms = 0.0;
};

inline TimingSummary summarize_timings(const std::vector<double> &samples) {
    if (samples.empty()) {
        return {};
    }

    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());

    double sum = 0.0;
    for (double value : samples) {
        sum += value;
    }

    const std::size_t middle = sorted.size() / 2;
    const double median = sorted.size() % 2 == 0
        ? (sorted[middle - 1] + sorted[middle]) * 0.5
        : sorted[middle];
    return {sorted.front(), median, sum / static_cast<double>(samples.size())};
}

inline BenchmarkReport run_benchmark(
    RenderBackend &backend,
    const SceneData &scene,
    const CameraSettings &camera,
    const RenderSettings &settings,
    const BenchmarkOptions &options,
    bool capture_last_sequence = false) {
    if (options.warmup_iterations < 0) {
        throw std::invalid_argument("warmup iteration count must be non-negative");
    }
    if (options.iterations < 1) {
        throw std::invalid_argument("iteration count must be at least one");
    }
    if (options.frames < 1) {
        throw std::invalid_argument("frame count must be at least one");
    }
    if (options.fps < 1) {
        throw std::invalid_argument("fps must be at least one");
    }

    const auto prepare_start = std::chrono::steady_clock::now();
    backend.prepare(scene, camera, settings);
    const auto prepare_end = std::chrono::steady_clock::now();

    BenchmarkReport report;
    report.prepare_ms = std::chrono::duration<double, std::milli>(
        prepare_end - prepare_start).count();

    for (int i = 0; i < options.warmup_iterations; ++i) {
        for (int frame = 0; frame < options.frames; ++frame) {
            backend.render(frame, options.frames);
        }
    }

    report.backend_samples_ms.reserve(static_cast<std::size_t>(options.iterations));
    report.wall_samples_ms.reserve(static_cast<std::size_t>(options.iterations));
    for (int i = 0; i < options.iterations; ++i) {
        const bool capture = capture_last_sequence && i + 1 == options.iterations;
        if (capture) {
            report.last_frames.clear();
            report.last_frames.reserve(static_cast<std::size_t>(options.frames));
        }

        double backend_ms = 0.0;
        const auto sequence_start = std::chrono::steady_clock::now();
        RenderFrame sequence_last_frame;
        for (int frame = 0; frame < options.frames; ++frame) {
            RenderFrame result = backend.render(frame, options.frames);
            backend_ms += result.timings.backend_ms;
            if (capture) {
                report.last_frames.emplace_back(std::move(result));
            } else if (frame + 1 == options.frames) {
                sequence_last_frame = std::move(result);
            }
        }
        const auto sequence_end = std::chrono::steady_clock::now();
        const double wall_ms = std::chrono::duration<double, std::milli>(
            sequence_end - sequence_start).count();
        report.backend_samples_ms.push_back(backend_ms);
        report.wall_samples_ms.push_back(wall_ms);
        if (capture) {
            report.last_frame = report.last_frames.back();
        } else {
            report.last_frame = std::move(sequence_last_frame);
        }
    }

    return report;
}

#endif
