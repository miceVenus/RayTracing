#include <cstdint>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <memory>
#include <sstream>

#include "../bench/benchmark.hpp"
#include "../bench/scene_factory.hpp"
#include "../core/scene.hpp"
#ifdef RT_ENABLE_CUDA
#include "../render/cuda/cuda_backend.hpp"
#endif
#include "../render/normal_cpu/cpu_backend.hpp"
#include "../render/ppm.hpp"
#ifdef RT_ENABLE_EMBREE
#include "../render/embree/embree_backend.hpp"
#endif
#ifdef RT_ENABLE_OPTIX
#include "../render/optix/optix_backend.hpp"
#endif

namespace {

struct CommandLineOptions {
    CameraSettings camera;
    RenderSettings render;
    BenchmarkOptions benchmark;
    std::string backend = "cpu";
    std::string format = "human";
    std::string output_path;
    std::string output_directory;
    double requested_duration_seconds = -1.0;
    bool duration_was_set = false;
    bool show_help = false;
};

std::string available_backends() {
    std::string result = "cpu";
#ifdef RT_ENABLE_CUDA
    result += ", cuda";
#endif
#ifdef RT_ENABLE_EMBREE
    result += ", embree";
#endif
#ifdef RT_ENABLE_OPTIX
    result += ", optix";
#endif
    return result;
}

std::unique_ptr<RenderBackend> make_backend(const std::string &name) {
    if (name == "cpu") {
        return std::make_unique<CpuBackend>();
    }
#ifdef RT_ENABLE_CUDA
    if (name == "cuda") {
        return std::make_unique<CudaBackend>();
    }
#endif
#ifdef RT_ENABLE_EMBREE
    if (name == "embree") {
        return std::make_unique<EmbreeBackend>();
    }
#endif
#ifdef RT_ENABLE_OPTIX
    if (name == "optix") {
        return std::make_unique<OptixBackend>();
    }
#endif
    throw std::invalid_argument("unsupported backend: " + name +
        " (available: " + available_backends() + ")");
}

void print_help(std::ostream &out) {
    out << "Ray tracing benchmark\n"
        << "Usage: raytracer_bench [options]\n\n"
        << "Options:\n"
        << "  --backend NAME       Renderer backend (available: " << available_backends() << ")\n"
        << "  --width N            Image width (default: 400)\n"
        << "  --aspect VALUE       Width / height (default: 16/9)\n"
        << "  --spp N              Samples per pixel (default: 500)\n"
        << "  --depth N            Maximum path depth (default: 50)\n"
        << "  --threads N          CPU worker threads (default: 16)\n"
        << "  --seed N             Fixed scene and sampling seed (default: 1)\n"
        << "  --warmup N           Unmeasured warmup renders (default: 0)\n"
        << "  --iterations N       Measured renders (default: 1)\n"
        << "  --frames N           Frames per sequence (default: 1)\n"
        << "  --fps N              Sequence frame rate (default: 30)\n"
        << "  --duration SECONDS   Set frame count from duration and fps\n"
        << "  --format FORMAT      human or csv (default: human)\n"
        << "  --output PATH        Write the last frame as a PPM image\n"
        << "  --output-dir DIR     Write every frame as numbered PPM images\n"
        << "                       Example: --frames 300 --fps 30 (10 seconds)\n"
        << "  --help               Show this help\n";
}

std::string require_value(int &index, int argc, char **argv, const std::string &option) {
    if (index + 1 >= argc) {
        throw std::invalid_argument("missing value for " + option);
    }
    return argv[++index];
}

int parse_int(const std::string &value, const std::string &option) {
    std::size_t consumed = 0;
    const long long parsed = std::stoll(value, &consumed);
    if (consumed != value.size() || parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("invalid integer for " + option + ": " + value);
    }
    return static_cast<int>(parsed);
}

double parse_double(const std::string &value, const std::string &option) {
    std::size_t consumed = 0;
    const double parsed = std::stod(value, &consumed);
    if (consumed != value.size()) {
        throw std::invalid_argument("invalid number for " + option + ": " + value);
    }
    if (!std::isfinite(parsed)) {
        throw std::invalid_argument("number for " + option + " must be finite");
    }
    return parsed;
}

std::uint64_t parse_seed(const std::string &value) {
    if (value.empty() || value.front() == '-') {
        throw std::invalid_argument("seed must be an unsigned integer");
    }
    std::size_t consumed = 0;
    const unsigned long long parsed = std::stoull(value, &consumed);
    if (consumed != value.size()) {
        throw std::invalid_argument("invalid seed: " + value);
    }
    return static_cast<std::uint64_t>(parsed);
}

CommandLineOptions parse_options(int argc, char **argv) {
    CommandLineOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--help" || option == "-h") {
            options.show_help = true;
        } else if (option == "--backend") {
            options.backend = require_value(i, argc, argv, option);
        } else if (option == "--width") {
            options.camera.image_width = parse_int(require_value(i, argc, argv, option), option);
        } else if (option == "--aspect") {
            options.camera.aspect_ratio = parse_double(require_value(i, argc, argv, option), option);
        } else if (option == "--spp") {
            options.render.samples_per_pixel = parse_int(require_value(i, argc, argv, option), option);
        } else if (option == "--depth") {
            options.render.max_depth = parse_int(require_value(i, argc, argv, option), option);
        } else if (option == "--threads") {
            options.render.thread_count = parse_int(require_value(i, argc, argv, option), option);
        } else if (option == "--seed") {
            options.render.seed = parse_seed(require_value(i, argc, argv, option));
        } else if (option == "--warmup") {
            options.benchmark.warmup_iterations = parse_int(require_value(i, argc, argv, option), option);
        } else if (option == "--iterations") {
            options.benchmark.iterations = parse_int(require_value(i, argc, argv, option), option);
        } else if (option == "--frames") {
            options.benchmark.frames = parse_int(require_value(i, argc, argv, option), option);
        } else if (option == "--fps") {
            options.benchmark.fps = parse_int(require_value(i, argc, argv, option), option);
        } else if (option == "--duration") {
            options.requested_duration_seconds = parse_double(
                require_value(i, argc, argv, option), option);
            options.duration_was_set = true;
        } else if (option == "--format") {
            options.format = require_value(i, argc, argv, option);
        } else if (option == "--output") {
            options.output_path = require_value(i, argc, argv, option);
        } else if (option == "--output-dir") {
            options.output_directory = require_value(i, argc, argv, option);
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }

    if (options.show_help) {
        return options;
    }
    if (options.camera.image_width < 1) {
        throw std::invalid_argument("image width must be positive");
    }
    if (!(options.camera.aspect_ratio > 0.0)) {
        throw std::invalid_argument("aspect ratio must be positive");
    }
    if (options.render.samples_per_pixel < 1 || options.render.max_depth < 1) {
        throw std::invalid_argument("samples per pixel and depth must be positive");
    }
    if (options.render.thread_count < 1) {
        throw std::invalid_argument("thread count must be positive");
    }
    if (options.benchmark.frames < 1 || options.benchmark.fps < 1) {
        throw std::invalid_argument("frames and fps must be positive");
    }
    if (options.duration_was_set) {
        if (options.requested_duration_seconds <= 0.0) {
            throw std::invalid_argument("duration must be positive");
        }
        const double exact_frames = options.requested_duration_seconds * options.benchmark.fps;
        if (!std::isfinite(exact_frames) ||
            exact_frames > static_cast<double>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("duration and fps exceed the supported frame count");
        }
        const auto rounded_frames = static_cast<long long>(std::llround(exact_frames));
        if (rounded_frames < 1 ||
            std::abs(exact_frames - static_cast<double>(rounded_frames)) > 1.0e-9 ||
            rounded_frames > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("duration times fps must be a positive integer frame count");
        }
        options.benchmark.frames = static_cast<int>(rounded_frames);
    }
    if (options.backend != "cpu"
#ifdef RT_ENABLE_CUDA
        && options.backend != "cuda"
#endif
#ifdef RT_ENABLE_EMBREE
        && options.backend != "embree"
#endif
#ifdef RT_ENABLE_OPTIX
        && options.backend != "optix"
#endif
    ) {
        throw std::invalid_argument("unsupported backend: " + options.backend +
            " (available: " + available_backends() + ")");
    }
    if (options.format != "human" && options.format != "csv") {
        throw std::invalid_argument("format must be human or csv");
    }
    return options;
}

void print_human_report(
    const CommandLineOptions &options,
    const SceneData &scene,
    const BenchmarkReport &report,
    std::string_view backend_name) {
    const TimingSummary backend = summarize_timings(report.backend_samples_ms);
    const TimingSummary wall = summarize_timings(report.wall_samples_ms);
    const double primary_samples = static_cast<double>(options.camera.image_width) *
        options.camera.image_height() * options.render.samples_per_pixel * options.benchmark.frames;
    const double samples_per_second = wall.mean_ms > 0.0
        ? primary_samples * 1000.0 / wall.mean_ms
        : 0.0;
    const double effective_fps = wall.mean_ms > 0.0
        ? options.benchmark.frames * 1000.0 / wall.mean_ms
        : 0.0;
    const double end_to_end_ms = report.prepare_ms + wall.mean_ms;
    const double cold_start_fps = end_to_end_ms > 0.0
        ? options.benchmark.frames * 1000.0 / end_to_end_ms
        : 0.0;

    std::cout << std::fixed << std::setprecision(3)
              << "backend: " << backend_name << '\n'
              << "scene: reference-spheres (" << scene.spheres.size() << " spheres)\n"
              << "image: " << options.camera.image_width << 'x' << options.camera.image_height() << '\n'
              << "sequence: " << options.benchmark.frames << " frames at " << options.benchmark.fps
              << " fps (" << static_cast<double>(options.benchmark.frames) / options.benchmark.fps
              << " seconds)\n"
              << "quality: " << options.render.samples_per_pixel << " spp, depth "
              << options.render.max_depth << '\n'
              << "seed: " << options.render.seed << ", threads: " << options.render.thread_count << '\n'
              << "prepare_ms: " << report.prepare_ms << '\n'
              << "render_backend_sequence_min_ms: " << backend.minimum_ms << '\n'
              << "render_backend_sequence_median_ms: " << backend.median_ms << '\n'
              << "render_backend_sequence_mean_ms: " << backend.mean_ms << '\n'
              << "render_backend_frame_mean_ms: " << backend.mean_ms / options.benchmark.frames << '\n'
              << "render_wall_sequence_mean_ms: " << wall.mean_ms << '\n'
              << "render_wall_frame_mean_ms: " << wall.mean_ms / options.benchmark.frames << '\n'
              << "effective_fps: " << effective_fps << '\n'
              << "prepare_plus_sequence_ms: " << end_to_end_ms << '\n'
              << "cold_start_effective_fps: " << cold_start_fps << '\n'
              << "primary_samples_per_second: " << samples_per_second << '\n';
}

void print_csv_report(
    const CommandLineOptions &options,
    const SceneData &scene,
    const BenchmarkReport &report,
    std::string_view backend_name) {
    const TimingSummary backend = summarize_timings(report.backend_samples_ms);
    const TimingSummary wall = summarize_timings(report.wall_samples_ms);
    const double primary_samples = static_cast<double>(options.camera.image_width) *
        options.camera.image_height() * options.render.samples_per_pixel * options.benchmark.frames;
    const double samples_per_second = wall.mean_ms > 0.0
        ? primary_samples * 1000.0 / wall.mean_ms
        : 0.0;

    const double effective_fps = wall.mean_ms > 0.0
        ? options.benchmark.frames * 1000.0 / wall.mean_ms
        : 0.0;
    const double end_to_end_ms = report.prepare_ms + wall.mean_ms;
    const double cold_start_fps = end_to_end_ms > 0.0
        ? options.benchmark.frames * 1000.0 / end_to_end_ms
        : 0.0;
    std::cout << "backend,scene,spheres,width,height,frames,fps,duration_seconds,spp,depth,seed,threads,warmup,iterations,prepare_ms,sequence_min_ms,sequence_median_ms,sequence_mean_ms,backend_frame_mean_ms,wall_sequence_mean_ms,wall_frame_mean_ms,effective_fps,prepare_plus_sequence_ms,cold_start_effective_fps,primary_samples_per_second\n"
              << std::fixed << std::setprecision(3)
              << backend_name << ",reference-spheres," << scene.spheres.size() << ','
              << options.camera.image_width << ',' << options.camera.image_height() << ','
              << options.benchmark.frames << ',' << options.benchmark.fps << ','
              << static_cast<double>(options.benchmark.frames) / options.benchmark.fps << ','
              << options.render.samples_per_pixel << ',' << options.render.max_depth << ','
              << options.render.seed << ',' << options.render.thread_count << ','
              << options.benchmark.warmup_iterations << ',' << options.benchmark.iterations << ','
              << report.prepare_ms << ',' << backend.minimum_ms << ',' << backend.median_ms << ','
              << backend.mean_ms << ',' << backend.mean_ms / options.benchmark.frames << ','
              << wall.mean_ms << ',' << wall.mean_ms / options.benchmark.frames << ','
              << effective_fps << ',' << end_to_end_ms << ',' << cold_start_fps << ','
              << samples_per_second << '\n';
}

} // namespace

int main(int argc, char **argv) {
    try {
        const CommandLineOptions options = parse_options(argc, argv);
        if (options.show_help) {
            print_help(std::cout);
            return 0;
        }

        const SceneData scene = benchmark_scene::make_reference_scene(options.render.seed);
        std::unique_ptr<RenderBackend> backend = make_backend(options.backend);
        const BenchmarkReport report = run_benchmark(
            *backend,
            scene,
            options.camera,
            options.render,
            options.benchmark,
            !options.output_directory.empty());

        if (options.format == "csv") {
            print_csv_report(options, scene, report, backend->name());
        } else {
            print_human_report(options, scene, report, backend->name());
        }

        if (!options.output_directory.empty()) {
            const std::filesystem::path directory(options.output_directory);
            std::filesystem::create_directories(directory);
            for (std::size_t frame_index = 0; frame_index < report.last_frames.size(); ++frame_index) {
                std::ostringstream filename;
                filename << "frame_" << std::setw(6) << std::setfill('0') << frame_index << ".ppm";
                write_ppm((directory / filename.str()).string(), report.last_frames[frame_index].image);
            }
            std::ostream &message_stream = options.format == "csv" ? std::cerr : std::cout;
            message_stream << "frames: " << report.last_frames.size() << " PPM images in "
                           << directory.string() << '\n';
        }

        if (!options.output_path.empty()) {
            write_ppm(options.output_path, report.last_frame.image);
            if (options.format == "csv") {
                std::cerr << "wrote image: " << options.output_path << '\n';
            } else {
                std::cout << "image: " << options.output_path << '\n';
            }
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n'
                  << "Use --help to see available options.\n";
        return 1;
    }
}
