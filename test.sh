#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: $0 {cpu|cuda|emtree|embree|optix}" >&2
}

if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi

case "$1" in
    cpu)
        backend="cpu"
        ;;
    emtree|embree)
        backend="embree"
        ;;
    optix)
        backend="optix"
        ;;
    cuda)
        backend="cuda"
        ;;
    *)
        usage
        exit 2
        ;;
esac

gpu_env_name=""
if [[ "$backend" == "cuda" || "$backend" == "optix" ]]; then
    gpu_env_name="${RAYTRACE_MICROMAMBA_ENV:-raytrace}"
    if [[ "${RT_GPU_MICROMAMBA_ACTIVE:-0}" != "1" ]]; then
        active_env_name="${CONDA_DEFAULT_ENV:-}"
        if [[ -z "$active_env_name" && -n "${CONDA_PREFIX:-}" ]]; then
            active_env_name="${CONDA_PREFIX##*/}"
        fi

        if [[ "$active_env_name" == "$gpu_env_name" ]]; then
            export RT_GPU_MICROMAMBA_ACTIVE=1
        else
            micromamba_executable="${MICROMAMBA_EXE:-$(type -P micromamba || true)}"
            if [[ -z "$micromamba_executable" ]]; then
                echo "The $backend backend requires micromamba environment '$gpu_env_name', but micromamba was not found." >&2
                echo "Set MICROMAMBA_EXE to the micromamba executable or activate that environment first." >&2
                exit 2
            fi

            exec "$micromamba_executable" run -n "$gpu_env_name" \
                --env RT_GPU_MICROMAMBA_ACTIVE=1 \
                bash "$project_root/test.sh" "$backend"
        fi
    fi
fi

if command -v nproc >/dev/null 2>&1; then
    logical_threads="$(nproc --all 2>/dev/null || nproc)"
elif command -v getconf >/dev/null 2>&1; then
    logical_threads="$(getconf _NPROCESSORS_ONLN)"
else
    logical_threads=1
fi

if [[ ! "$logical_threads" =~ ^[1-9][0-9]*$ ]]; then
    echo "Could not determine the number of logical CPU threads." >&2
    exit 2
fi

render_threads=$((logical_threads * 2 / 3))
if (( render_threads < 1 )); then
    render_threads=1
fi

build_dir="$project_root/build-benchmark/$backend"
cmake_command="cmake"
cmake_args=(
    -S "$project_root"
    -B "$build_dir"
    -DCMAKE_BUILD_TYPE=Release
)

case "$backend" in
    embree)
        cmake_args+=(-DRT_ENABLE_EMBREE=ON)
        if [[ -n "${CONDA_PREFIX:-}" ]]; then
            cmake_args+=("-DCMAKE_PREFIX_PATH=${CONDA_PREFIX}")
        fi
        ;;
    cuda|optix)
        if [[ -z "${CONDA_PREFIX:-}" ]]; then
            echo "The $backend backend must be built inside micromamba environment '$gpu_env_name'." >&2
            exit 2
        fi

        cmake_command="${CONDA_PREFIX}/bin/cmake"
        cuda_compiler="${CONDA_PREFIX}/bin/nvcc"
        if [[ ! -x "$cmake_command" ]]; then
            echo "CMake was not found in the active environment: $cmake_command" >&2
            exit 2
        fi
        if [[ ! -x "$cuda_compiler" ]]; then
            echo "nvcc was not found in micromamba environment '$gpu_env_name': $cuda_compiler" >&2
            echo "Install the CUDA compiler/runtime development packages into this environment:" >&2
            echo "  micromamba install -n '$gpu_env_name' -c nvidia/label/cuda-12.3.2 cuda-nvcc cuda-cudart-dev cuda-driver-dev" >&2
            exit 2
        fi

        cmake_args+=("-DCMAKE_CUDA_COMPILER=${cuda_compiler}")
        cmake_args+=("-DCUDAToolkit_ROOT=${CONDA_PREFIX}")
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

        if [[ "$backend" == "cuda" ]]; then
            cmake_args+=(-DRT_ENABLE_CUDA=ON)
            cmake_args+=("-DRT_CUDA_ARCHITECTURES=${RT_CUDA_ARCHITECTURES:-86}")
        else
            cmake_args+=(-DRT_ENABLE_OPTIX=ON)
            optix_root_dir="${OPTIX_ROOT_DIR:-${OPTIX_ROOT:-}}"
            if [[ -n "$optix_root_dir" ]]; then
                cmake_args+=("-DOPTIX_ROOT_DIR=${optix_root_dir}")
                cmake_args+=("-DOPTIX_INCLUDE_DIR=${optix_root_dir}/include")
            fi
        fi
        ;;
esac

echo "Configuring ${backend} benchmark (host threads: ${render_threads}/${logical_threads})"
if [[ "$backend" == "cuda" || "$backend" == "optix" ]]; then
    echo "CUDA toolchain: CMake=${cmake_command}, nvcc=${cuda_compiler}"
fi
"$cmake_command" "${cmake_args[@]}"
"$cmake_command" --build "$build_dir" --parallel "$render_threads"

run_id="$(date +%Y%m%d_%H%M%S)_$$"
frames_dir="$project_root/test/frames-${backend}-${run_id}"
mkdir -p "$frames_dir"

render_args=(
    --backend "$backend"
    --width 400
    --aspect 1.7777777777777777
    --duration 10
    --fps 30
    --spp 100
    --depth 50
    --seed 1
    --warmup 1
    --iterations 3
    --threads "$render_threads"
    --format csv
    --output-dir "$frames_dir"
)

echo "Rendering ${backend}: 400x225, 300 frames, 100 spp, depth 50, seed 1"
echo "PPM frames and timing CSV: ${frames_dir}"
"$build_dir/raytracer_bench" "${render_args[@]}" | tee "$frames_dir/timing.csv"
