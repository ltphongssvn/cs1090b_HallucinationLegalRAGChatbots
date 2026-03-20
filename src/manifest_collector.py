# src/manifest_collector.py
import argparse
import contextlib
import importlib.metadata as meta
import json
import multiprocessing
import os
import platform
import subprocess
import sys
from datetime import datetime


def get_nvcc_version() -> str:
    try:
        r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if "release" in line:
                return line.strip()
    except FileNotFoundError:
        return "nvcc not found"
    return "unknown"


def get_driver_cuda() -> str:
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if "CUDA Version" in line:
                return line.strip().split()[-1]
    except FileNotFoundError:
        return "nvidia-smi not found"
    return "unknown"


def get_driver_version() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        return r.stdout.strip().splitlines()[0] if r.stdout.strip() else "unknown"
    except FileNotFoundError:
        return "nvidia-smi not found"


def get_faiss_version() -> str:
    try:
        import faiss  # type: ignore[import]

        return getattr(faiss, "__version__", "installed — no version attr")
    except ImportError:
        return "not installed"


def get_installed_versions(pkgs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in pkgs:
        try:
            out[p] = meta.version(p)
        except meta.PackageNotFoundError:
            out[p] = "not installed"
    return out


def parse_freeze(freeze_str: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in freeze_str.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" in line:
            pkg, _, ver = line.partition("==")
            result[pkg.strip()] = ver.strip()
        else:
            result[line] = "unknown-format"
    return result


def get_cpu_info() -> dict[str, object]:
    info: dict[str, object] = {
        "physical_cores": "unknown",
        "logical_cores": "unknown",
        "cpu_model": "unknown",
        "ram_total_gb": "unknown",
    }
    try:
        info["logical_cores"] = multiprocessing.cpu_count()
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            lines = f.read().splitlines()
        models = [ln.split(":")[1].strip() for ln in lines if ln.startswith("model name")]
        if models:
            info["cpu_model"] = models[0]
        physical = len({ln.split(":")[1].strip() for ln in lines if ln.startswith("physical id")})
        cores_per = len({ln.split(":")[1].strip() for ln in lines if ln.startswith("core id")})
        if physical > 0 and cores_per > 0:
            info["physical_cores"] = physical * cores_per
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    info["ram_total_gb"] = round(kb / 1024 / 1024, 1)
                    break
    except Exception:
        pass
    return info


def get_gpu_list() -> list[dict[str, object]]:
    import torch  # type: ignore[import]

    gpus: list[dict[str, object]] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(
                {
                    "index": i,
                    "name": props.name,
                    "vram_gb": round(props.total_memory / 1e9, 2),
                    "compute_capability": list(torch.cuda.get_device_capability(i)),
                }
            )
    return gpus


def collect(args: argparse.Namespace) -> dict[str, object]:
    import spacy  # type: ignore[import]
    import torch  # type: ignore[import]
    import transformers  # type: ignore[import]

    # Apply reproducibility settings before collecting stability flags.
    # Redirect stdout to stderr — manifest_collector prints pure JSON to stdout;
    # configure() prints human-readable status which must not pollute that stream.
    try:
        from src.repro import configure  # type: ignore[import]

        with contextlib.redirect_stdout(sys.stderr):
            configure()
    except Exception:
        pass  # repro.py may not exist on first run before write_repro_module

    freeze_parsed = parse_freeze(args.freeze) if args.freeze not in ("unavailable", "") else {}
    gpus = get_gpu_list()
    nlp = spacy.load(args.spacy_model)
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_sha": args.git_sha,
        "git_branch": args.git_branch,
        "git_dirty_files": int(args.git_dirty) if args.git_dirty.isdigit() else -1,
        "uv_lock_sha256": args.uvlock_sha256,
        "uv_version": args.uv_version,
        "hostname": args.hostname,
        "slurm_job_id": args.slurm_job_id,
        "slurm_job_name": args.slurm_job_name,
        "slurm_nodelist": args.slurm_nodelist,
        "cpu": get_cpu_info(),
        "python": sys.version,
        "python_path": sys.path,
        "platform": platform.platform(),
        "cuda_env": {
            "CUDA_HOME": os.environ.get("CUDA_HOME", "NOT SET"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "NOT SET"),
        },
        "repro_env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "NOT SET"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "NOT SET"),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "NOT SET"),
            "RANDOM_SEED": os.environ.get("RANDOM_SEED", "NOT SET"),
        },
        "numerical_stability": {
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        },
        "parity_module": "src/repro.py",
        "parity_usage": "from src.repro import configure; configure()",
        "hardware_target": {
            "gpu_name": args.target_gpu_name,
            "gpu_count": int(args.target_gpu_count),
            "compute_cap_min": [int(args.target_cap_major), int(args.target_cap_minor)],
            "vram_gb_min": float(args.target_vram_gb_min),
            "torch_cuda_runtime": args.target_torch_cuda,
            "driver_cuda": args.target_driver_cuda,
            "python_version": args.target_python_version,
            "min_disk_gb": int(args.target_min_disk_gb),
        },
        "hardware_detected": {
            "gpu_name": args.detected_gpu_name,
            "gpu_count": args.detected_gpu_count,
            "torch_cuda_runtime": args.detected_torch_cuda,
            "driver_cuda": args.detected_driver_cuda,
            "cudnn": args.detected_cudnn,
            "hardware_match": args.hardware_match,
        },
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "driver_cuda": get_driver_cuda(),
        "driver_version": get_driver_version(),
        "cudnn": str(torch.backends.cudnn.version()) if torch.cuda.is_available() else None,
        "cuda_toolkit_nvcc": get_nvcc_version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": gpus,
        "transformers": transformers.__version__,
        "spacy": spacy.__version__,
        "spacy_model": args.spacy_model,
        "spacy_model_version": nlp.meta.get("version"),
        "spacy_model_sha256": args.spacy_model_sha256,
        "faiss": get_faiss_version(),
        "installed_packages": get_installed_versions(
            [
                "torch",
                "transformers",
                "datasets",
                "faiss-cpu",
                "spacy",
                "scikit-learn",
                "numpy",
                "pandas",
                "langchain",
                "gensim",
                "sentence-transformers",
                "networkx",
                "accelerate",
                "peft",
                "evaluate",
                "ragas",
                "rouge-score",
                "wandb",
                "pytest",
                "mypy",
                "hypothesis",
            ]
        ),
        "freeze_snapshot": freeze_parsed,
        "freeze_snapshot_raw": args.freeze,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect environment manifest and print JSON.")
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--git-branch", required=True)
    parser.add_argument("--git-dirty", required=True)
    parser.add_argument("--uvlock-sha256", required=True)
    parser.add_argument("--uv-version", default="unknown")
    parser.add_argument("--hostname", default="unknown")
    parser.add_argument("--slurm-job-id", default="none")
    parser.add_argument("--slurm-job-name", default="none")
    parser.add_argument("--slurm-nodelist", default="none")
    parser.add_argument("--target-gpu-name", required=True)
    parser.add_argument("--target-gpu-count", required=True)
    parser.add_argument("--target-cap-major", required=True)
    parser.add_argument("--target-cap-minor", required=True)
    parser.add_argument("--target-vram-gb-min", required=True)
    parser.add_argument("--target-torch-cuda", required=True)
    parser.add_argument("--target-driver-cuda", required=True)
    parser.add_argument("--target-python-version", required=True)
    parser.add_argument("--target-min-disk-gb", required=True)
    parser.add_argument("--detected-gpu-name", required=True)
    parser.add_argument("--detected-gpu-count", required=True)
    parser.add_argument("--detected-torch-cuda", required=True)
    parser.add_argument("--detected-driver-cuda", required=True)
    parser.add_argument("--detected-cudnn", required=True)
    parser.add_argument("--hardware-match", required=True)
    parser.add_argument("--spacy-model", required=True)
    parser.add_argument("--spacy-model-sha256", required=True)
    parser.add_argument("--freeze", default="unavailable")
    args = parser.parse_args()
    data = collect(args)
    print(json.dumps(data))


if __name__ == "__main__":
    main()
