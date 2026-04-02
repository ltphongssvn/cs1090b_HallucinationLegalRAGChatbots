# tests/test_environment_artifacts.py
# Artifact verification — venv/manifest/kernelspec checks.
# TestVenv: marked unit (venv always present after setup)
# TestManifest / TestKernelspec: marked artifact — require setup.sh to have run
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
MANIFEST = PROJECT_ROOT / "logs" / "environment_manifest.json"
KERNELSPEC_DIR = Path.home() / ".local" / "share" / "jupyter" / "kernels" / "hallucination-legal-rag"


def _venv_run(code: str) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
    return subprocess.run([str(VENV_PYTHON), "-c", code], capture_output=True, text=True)


skip_no_venv = pytest.mark.skipif(not VENV_PYTHON.is_file(), reason=".venv not built — run bash setup.sh first")
skip_no_manifest = pytest.mark.skipif(not MANIFEST.is_file(), reason="manifest not written — run bash setup.sh first")
skip_no_env = pytest.mark.skipif(
    not (PROJECT_ROOT / ".env").is_file(), reason=".env not found — run bash setup.sh first"
)
skip_no_kernelspec = pytest.mark.skipif(
    not KERNELSPEC_DIR.is_dir(), reason="kernelspec not registered — run bash setup.sh first"
)


@pytest.mark.unit
class TestVenv:
    @skip_no_venv
    def test_python_executable_exists(self) -> None:
        assert VENV_PYTHON.is_file() and os.access(VENV_PYTHON, os.X_OK)

    @skip_no_venv
    def test_python_version_is_3_11_9(self) -> None:
        r = _venv_run("import sys; print('.'.join(map(str,sys.version_info[:3])))")
        assert r.returncode == 0 and r.stdout.strip() == "3.11.9"

    @skip_no_venv
    def test_python_executable_inside_project_root(self) -> None:
        r = _venv_run("import sys; print(sys.executable)")
        assert r.returncode == 0 and str(PROJECT_ROOT / ".venv") in r.stdout

    @skip_no_venv
    def test_torch_version_is_cu117(self) -> None:
        r = _venv_run("import torch; print(torch.__version__)")
        assert r.returncode == 0
        assert "2.0.1" in r.stdout and "cu117" in r.stdout
        assert "cpu" not in r.stdout

    @skip_no_venv
    def test_torch_cuda_runtime_is_11_7(self) -> None:
        r = _venv_run("import torch; print(torch.version.cuda)")
        assert r.returncode == 0 and "11.7" in r.stdout

    @skip_no_venv
    def test_transformers_version_in_range(self) -> None:
        r = _venv_run("""
from packaging.version import Version
import transformers
v = Version(transformers.__version__)
assert Version('4.35') <= v < Version('4.41'), f'out of range: {v}'
print('ok')
""")
        assert r.returncode == 0

    @skip_no_venv
    def test_faiss_functional(self) -> None:
        r = _venv_run("""
import faiss, numpy as np
idx = faiss.IndexFlatL2(8)
vecs = np.random.rand(5, 8).astype('float32')
idx.add(vecs)
D, I = idx.search(vecs[:1], 2)
assert I.shape == (1, 2)
print('ok')
""")
        assert r.returncode == 0 and r.stdout.strip() == "ok"

    @skip_no_venv
    def test_spacy_model_version(self) -> None:
        r = _venv_run("""
import spacy
nlp = spacy.load('en_core_web_sm')
v = nlp.meta.get('version')
assert v == '3.8.0', f'expected 3.8.0, got {v}'
print(v)
""")
        assert r.returncode == 0 and r.stdout.strip() == "3.8.0"

    @skip_no_venv
    def test_spacy_ner_on_legal_text(self) -> None:
        r = _venv_run("""
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp('The Supreme Court ruled in favor of the plaintiff in Smith v. Jones.')
ents = [e.text for e in doc.ents]
assert len(ents) > 0
print(ents)
""")
        assert r.returncode == 0 and "Supreme Court" in r.stdout

    @skip_no_venv
    def test_dev_tools_installed(self) -> None:
        r = _venv_run("import pytest, mypy, hypothesis; print('ok')")
        assert r.returncode == 0 and r.stdout.strip() == "ok"

    @skip_no_venv
    def test_ipykernel_installed(self) -> None:
        r = _venv_run("import ipykernel; print(ipykernel.__version__)")
        assert r.returncode == 0 and r.stdout.strip()


# TestManifest and TestKernelspec are NOT marked unit — they depend on
# runtime artifacts written by setup.sh and must not run in the pre-commit
# fast gate. Run explicitly: pytest tests/test_environment_artifacts.py -m artifact
@pytest.mark.artifact
class TestManifest:
    def _load(self) -> dict:  # type: ignore[type-arg]
        return json.loads(MANIFEST.read_text())

    @skip_no_manifest
    def test_manifest_is_valid_json(self) -> None:
        assert isinstance(self._load(), dict)

    @skip_no_manifest
    def test_manifest_required_keys(self) -> None:
        data = self._load()
        required = [
            "timestamp",
            "git_sha",
            "git_branch",
            "git_dirty_files",
            "uv_lock_sha256",
            "python",
            "platform",
            "repro_env",
            "numerical_stability",
            "parity_module",
            "hardware_target",
            "hardware_detected",
            "torch",
            "torch_cuda_runtime",
            "cuda_available",
            "gpu_count",
            "gpus",
            "transformers",
            "spacy",
            "spacy_model",
            "spacy_model_version",
            "faiss",
            "installed_packages",
        ]
        missing = [k for k in required if k not in data]
        assert not missing, f"Missing keys: {missing}"

    @skip_no_manifest
    def test_no_undetected_sentinels(self) -> None:
        assert "UNDETECTED" not in MANIFEST.read_text()

    @skip_no_manifest
    def test_repro_env_no_not_set(self) -> None:
        bad = {k: v for k, v in self._load()["repro_env"].items() if v == "NOT SET"}
        assert not bad, f"NOT SET keys: {list(bad)}"

    @skip_no_manifest
    def test_repro_env_pythonhashseed_is_0(self) -> None:
        assert self._load()["repro_env"]["PYTHONHASHSEED"] == "0"

    @skip_no_manifest
    def test_repro_env_random_seed_is_0(self) -> None:
        assert self._load()["repro_env"]["RANDOM_SEED"] == "0"

    @skip_no_manifest
    def test_deterministic_algorithms_enabled(self) -> None:
        # torch.use_deterministic_algorithms(True) is set by repro.configure()
        # before manifest_collector runs. Value is True when setup.sh is run fresh.
        assert self._load()["numerical_stability"]["deterministic_algorithms_enabled"] is True

    @skip_no_manifest
    def test_cudnn_benchmark_is_false(self) -> None:
        assert self._load()["numerical_stability"]["cudnn_benchmark"] is False

    @skip_no_manifest
    def test_hardware_match_is_true(self) -> None:
        v = self._load()["hardware_detected"]["hardware_match"]
        assert str(v).lower() == "true"

    @skip_no_manifest
    def test_gpu_count_is_4(self) -> None:
        assert int(self._load()["hardware_detected"]["gpu_count"]) == 4

    @skip_no_manifest
    def test_gpu_count_matches_gpus_array(self) -> None:
        data = self._load()
        assert int(data["gpu_count"]) == len(data["gpus"])

    @skip_no_manifest
    def test_torch_version_in_manifest(self) -> None:
        v = self._load()["torch"]
        assert "2.0.1" in v and "cu117" in v

    @skip_no_manifest
    def test_uv_lock_sha256_matches_disk(self) -> None:
        recorded = self._load()["uv_lock_sha256"]
        actual = hashlib.sha256((PROJECT_ROOT / "uv.lock").read_bytes()).hexdigest()
        assert recorded == actual

    @skip_no_manifest
    def test_git_sha_not_sentinel(self) -> None:
        assert self._load()["git_sha"] != "not-a-git-repo"

    @skip_no_manifest
    def test_spacy_model_version_is_3_8_0(self) -> None:
        assert self._load()["spacy_model_version"] == "3.8.0"

    @skip_no_manifest
    def test_core_packages_installed(self) -> None:
        pkgs = self._load().get("installed_packages", {})
        core = ["torch", "transformers", "datasets", "faiss-cpu", "spacy", "numpy", "pandas"]
        bad = {k: pkgs.get(k) for k in core if pkgs.get(k) == "not installed"}
        assert not bad, f"Missing: {bad}"

    @skip_no_manifest
    def test_parity_module_is_repro(self) -> None:
        assert "repro" in self._load().get("parity_module", "").lower()


@pytest.mark.artifact
class TestKernelspec:
    def _load(self) -> dict:  # type: ignore[type-arg]
        return json.loads((KERNELSPEC_DIR / "kernel.json").read_text())

    @skip_no_kernelspec
    def test_kernelspec_dir_exists(self) -> None:
        assert KERNELSPEC_DIR.is_dir()

    @skip_no_kernelspec
    def test_kernel_json_valid(self) -> None:
        assert isinstance(self._load(), dict)

    @skip_no_kernelspec
    def test_display_name_contains_version(self) -> None:
        assert "3.11.9" in self._load().get("display_name", "")

    @skip_no_kernelspec
    def test_argv0_in_venv(self) -> None:
        assert ".venv" in self._load().get("argv", [""])[0]

    @skip_no_kernelspec
    def test_argv0_executable(self) -> None:
        argv0 = Path(self._load().get("argv", [""])[0])
        assert argv0.is_file() and os.access(argv0, os.X_OK)

    @skip_no_kernelspec
    def test_argv0_python_version(self) -> None:
        argv0 = self._load().get("argv", [""])[0]
        r = subprocess.run([argv0, "--version"], capture_output=True, text=True)
        assert "3.11.9" in (r.stdout + r.stderr)

    @skip_no_kernelspec
    def test_language_is_python(self) -> None:
        assert self._load().get("language") == "python"


@pytest.mark.unit
class TestFailurePaths:
    """Python-level failure path tests migrated from test_failure_paths.bats."""

    @skip_no_venv
    def test_torch_is_cuda_build_not_cpu(self) -> None:
        r = _venv_run("import torch; v=torch.__version__; assert 'cu' in v, f'CPU wheel: {v}'; print('ok')")
        assert r.returncode == 0

    @skip_no_venv
    def test_faiss_index_search_shape(self) -> None:
        r = _venv_run("""
import faiss, numpy as np
idx = faiss.IndexFlatL2(8)
vecs = np.random.rand(5, 8).astype('float32')
idx.add(vecs)
D, I = idx.search(vecs[:1], 2)
assert I.shape == (1, 2), f'bad shape: {I.shape}'
print('ok')
""")
        assert r.returncode == 0

    @skip_no_venv
    def test_spacy_model_wrong_version_fails(self) -> None:
        r = _venv_run("""
import spacy, sys
nlp = spacy.load('en_core_web_sm')
v = nlp.meta.get('version')
if v != '3.8.0':
    print(f'wrong version: {v}')
    sys.exit(1)
print('ok')
""")
        assert r.returncode == 0

    @skip_no_venv
    @skip_no_env
    def test_repro_configure_sets_deterministic(self) -> None:
        r = _venv_run("""
import sys
sys.path.insert(0, '.')
from src.repro import configure
configure()
import torch
assert torch.are_deterministic_algorithms_enabled(), 'deterministic not enabled'
print('ok')
""")
        assert r.returncode == 0
