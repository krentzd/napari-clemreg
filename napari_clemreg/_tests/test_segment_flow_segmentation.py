"""Segment-Flow EM segmentation integration.

Most of this is testable without Nextflow: prerequisite checking and the
revision pin. The real end-to-end test needs an actual Nextflow + Conda
install (and, on some systems, a specific JDK -- see the module
docstring), so it's marked slow and additionally skipped outright if
those tools aren't on PATH, rather than failing CI (which has neither).
"""
import shutil
from unittest.mock import patch

import numpy as np
import tifffile
import pytest

from napari_clemreg.clemreg.segment_flow_segmentation import (
    SEGMENT_FLOW_REPO,
    SEGMENT_FLOW_REVISION,
    SegmentFlowNotAvailable,
    _build_model_config,
    _build_subprocess_env,
    _check_prerequisites,
    _default_profile,
    _segment_flow_source,
    segment_flow_em_segmentation,
)

_HAS_NEXTFLOW = shutil.which("nextflow") is not None and shutil.which("conda") is not None
try:
    import aiod_registry  # noqa: F401
    _HAS_AIOD_REGISTRY = True
except ImportError:
    _HAS_AIOD_REGISTRY = False


def test_segment_flow_revision_is_pinned_not_default_branch():
    # Must not be "master"/"main" (the default branch) -- see the module
    # docstring for why: running the bare default branch fails in ways
    # that disappear once pinned to the same revision aiod_napari uses.
    assert SEGMENT_FLOW_REVISION not in ("master", "main")
    assert SEGMENT_FLOW_REPO == "FrancisCrickInstitute/Segment-Flow"


def test_segment_flow_source_defaults_to_pinned_github_release(monkeypatch):
    monkeypatch.delenv("AIOD_NXF_REPO", raising=False)
    assert _segment_flow_source() == [SEGMENT_FLOW_REPO, "-r", SEGMENT_FLOW_REVISION]


def test_segment_flow_source_uses_local_checkout_without_revision(monkeypatch, tmp_path):
    monkeypatch.setenv("AIOD_NXF_REPO", str(tmp_path))
    assert _segment_flow_source() == [str(tmp_path)]


def test_segment_flow_source_rejects_missing_local_checkout(monkeypatch, tmp_path):
    monkeypatch.setenv("AIOD_NXF_REPO", str(tmp_path / "missing"))
    with pytest.raises(SegmentFlowNotAvailable, match="AIOD_NXF_REPO"):
        _segment_flow_source()


def test_default_profile_honours_env(monkeypatch):
    monkeypatch.delenv("AIOD_NXF_PROFILE", raising=False)
    assert _default_profile() == "local"
    monkeypatch.setenv("AIOD_NXF_PROFILE", "berta")
    assert _default_profile() == "berta"


def test_check_prerequisites_raises_clearly_when_nextflow_missing():
    with patch("shutil.which", return_value=None):
        with pytest.raises(SegmentFlowNotAvailable, match="nextflow"):
            _check_prerequisites()


def test_check_prerequisites_passes_when_tools_present():
    with patch("shutil.which", return_value="/usr/bin/fake"):
        _check_prerequisites()  # should not raise


def test_segment_flow_em_segmentation_raises_clearly_without_prerequisites():
    with patch("shutil.which", return_value=None):
        with pytest.raises(SegmentFlowNotAvailable):
            segment_flow_em_segmentation(np.zeros((2, 2, 2), dtype=np.uint8))


@pytest.mark.skipif(not _HAS_AIOD_REGISTRY, reason="needs the aiod_registry package")
def test_build_model_config_patches_only_conf_threshold(tmp_path):
    """aiod_registry's own default conf_threshold for MitoNet v1 (0.5) is
    notably stricter than empanada-dl's own bundled config (0.3, see
    empanada_configs/MitoNet_V1.yaml) -- confirmed directly by generating
    and reading Segment-Flow's actual default config, and a real
    production run against a real EM volume (known, via the bundled
    empanada-dl path, to contain mitochondria) found none at all via
    Segment-Flow. This patches just that one field onto Segment-Flow's
    own generated default (not a hand-written config, which would risk
    an incomplete/wrong schema).
    """
    config_path = _build_model_config("MitoNet v1", "mito", 0.3, tmp_path)

    import yaml
    config = yaml.safe_load(config_path.read_text())

    assert config["conf_threshold"] == 0.3
    # Confirm this really did come from the registry's own generator, not
    # a hand-written stand-in -- these are AIoD's field names, not ours.
    assert config["plane"] == "XY"
    assert config["center_threshold"] == 0.1
    assert config["min_distance"] == 3


def test_build_subprocess_env_strips_venv_from_path():
    """`uv run` (and a plain venv activation) prepends the venv's own
    bin/ to PATH and sets VIRTUAL_ENV, *without* deactivating an
    already-active conda env. Confirmed directly this breaks Nextflow's
    own per-process conda activation for entirely unrelated envs (e.g.
    setupModel's, which needs aiod_registry): conda's activation script
    has no knowledge of the venv's bin/ entry and never removes it, so
    it keeps winning over whatever conda env conda just "activated" --
    `python` (and everything else) kept resolving to the venv's own
    interpreter regardless. This was previously misdiagnosed as a
    transient concurrency race (see git history); the real cause is
    deterministic and is this.
    """
    fake_venv = "/Users/fake/some-project/.venv"
    fake_env = {
        "VIRTUAL_ENV": fake_venv,
        "PATH": f"{fake_venv}/bin:/usr/bin:/bin",
    }
    with patch("os.environ", fake_env):
        env = _build_subprocess_env()

    assert "VIRTUAL_ENV" not in env
    assert f"{fake_venv}/bin" not in env["PATH"].split(":")
    assert env["PATH"] == "/usr/bin:/bin"


def test_build_subprocess_env_leaves_path_alone_without_a_venv():
    with patch("os.environ", {"PATH": "/usr/bin:/bin"}):
        env = _build_subprocess_env()

    assert env["PATH"] == "/usr/bin:/bin"


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_NEXTFLOW, reason="needs a real Nextflow + Conda install")
def test_segment_flow_em_segmentation_real_run(tmp_path):
    """Runs the actual pipeline against synthetic noise (no real
    mitochondria expected -- this checks the round trip works and
    returns a correctly-shaped mask, not segmentation quality).
    """
    rng = np.random.default_rng(0)
    volume = rng.integers(0, 255, size=(8, 64, 64), dtype=np.uint8)

    mask = segment_flow_em_segmentation(volume, root_dir=tmp_path)

    assert mask.shape == volume.shape


# --- Which mask a run returns -------------------------------------------------
#
# Segment-Flow writes every run's mask to the same cache folder, as
# <image>_masks_<hash>_all.tiff. Unless given --param_hash, it hashes all
# its parameters, including a temporary path, so each run's mask gets an
# unrelated name. A fake Nextflow run writes the mask the same way.

def _mask_dir(root):
    return root / "aiod_cache" / "empanada" / "MitoNet v1_masks"


@pytest.fixture
def fake_nextflow(monkeypatch, tmp_path):
    """Run segment_flow_em_segmentation without Nextflow. Its mask is the
    input volume plus 1. The cache already holds an older run's mask, which
    sorts after any other name. Returns the cache root."""
    import secrets
    from napari_clemreg.clemreg import segment_flow_segmentation as sf

    root = tmp_path / "aiod"
    _mask_dir(root).mkdir(parents=True)
    tifffile.imwrite(_mask_dir(root) / "input_tif_masks_ffffffff_all.tiff", np.full((2, 3, 4), 7, np.uint16))

    def run(cmd, cwd, env):
        param_hash = cmd[cmd.index("--param_hash") + 1] if "--param_hash" in cmd else secrets.token_hex(4)
        volume = tifffile.imread(cmd[cmd.index("--img_dir") + 1].replace("images.csv", "input.tif"))
        tifffile.imwrite(_mask_dir(root) / f"input_tif_masks_{param_hash}_all.tiff", volume + 1)
        # splitStacks' list of sub-stacks, also named by the hash.
        (root / "aiod_cache" / "splits").mkdir(exist_ok=True)
        (root / "aiod_cache" / "splits" / f"substacks_{param_hash}.csv").write_text("z")
        return 0, ""

    monkeypatch.setattr(sf, "_check_prerequisites", lambda: None)
    monkeypatch.setattr(sf, "_run_nextflow", run)
    monkeypatch.setattr(sf, "_build_model_config", lambda *args: tmp_path / "model_config.yml")
    return root


def test_a_run_returns_its_own_mask_not_an_older_one(fake_nextflow):
    volume = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    mask = segment_flow_em_segmentation(volume, root_dir=fake_nextflow)
    np.testing.assert_array_equal(mask, volume + 1)


def test_a_run_deletes_its_files_once_read_and_leaves_others(fake_nextflow):
    segment_flow_em_segmentation(np.zeros((2, 3, 4), np.uint16), root_dir=fake_nextflow)
    assert [p.name for p in _mask_dir(fake_nextflow).iterdir()] == ["input_tif_masks_ffffffff_all.tiff"]
    assert list((fake_nextflow / "aiod_cache" / "splits").iterdir()) == []


def test_a_run_without_its_mask_fails_instead_of_using_another(fake_nextflow, monkeypatch):
    from napari_clemreg.clemreg import segment_flow_segmentation as sf

    monkeypatch.setattr(sf, "_run_nextflow", lambda cmd, cwd, env: (0, ""))
    with pytest.raises(sf.SegmentFlowRunError, match="no output mask"):
        segment_flow_em_segmentation(np.zeros((2, 3, 4), np.uint16), root_dir=fake_nextflow)


# --- CLEM-Reg's additions to Segment-Flow's Nextflow config -------------------

@pytest.fixture
def nextflow_runs(monkeypatch, tmp_path):
    """Start segment_flow_em_segmentation without Nextflow. Returns a list
    that gets each run's command and the text of its -c config file."""
    from napari_clemreg.clemreg import segment_flow_segmentation as sf

    runs = []

    def run(cmd, cwd, env):
        config = cmd[cmd.index("-c") + 1] if "-c" in cmd else None
        runs.append((cmd, open(config).read() if config else None))
        return 1, "stopped by the test"

    monkeypatch.setattr(sf, "_check_prerequisites", lambda: None)
    monkeypatch.setattr(sf, "_run_nextflow", run)
    monkeypatch.setattr(sf, "_build_model_config", lambda *args: tmp_path / "model_config.yml")
    return runs


@pytest.mark.parametrize("profile, one_at_a_time", [("local", True), ("crick", False)])
def test_run_model_gets_clemregs_nextflow_config(nextflow_runs, tmp_path, profile, one_at_a_time):
    """On a local machine runModel runs one task at a time, as Segment-Flow's
    own single-GPU profile does, and keeps LD_LIBRARY_PATH."""
    from napari_clemreg.clemreg.segment_flow_segmentation import SegmentFlowRunError

    with pytest.raises(SegmentFlowRunError):
        segment_flow_em_segmentation(np.zeros((2, 3, 4), np.uint8), profile=profile, root_dir=tmp_path)

    (cmd, config), = nextflow_runs
    assert ("withName: 'runModel'" in config) == one_at_a_time
    assert ("maxForks = 1" in config) == one_at_a_time
    assert "LD_LIBRARY_PATH" not in config


_TOY_PIPELINE = '''
process runModel {
    label 'gpu_process'
    input: val x
    output: stdout
    script:
    """
    echo "LD=\\${LD_LIBRARY_PATH:-unset} start=\\$(python3 -c 'import time; print(time.time())')"
    sleep 1
    echo "end=\\$(python3 -c 'import time; print(time.time())')"
    """
}
workflow { runModel(Channel.of(1, 2, 3)).view() }
'''
# Like Segment-Flow's: a local profile giving runModel 4 CPUs, so several
# tasks would otherwise run at once.
_TOY_CONFIG = "profiles { local { process { withLabel: gpu_process { cpus = 1 } } } }\n"


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_NEXTFLOW, reason="needs a real Nextflow install")
def test_clemregs_nextflow_config_works_in_real_nextflow(tmp_path):
    """A runModel task sees the LD_LIBRARY_PATH it was started with (in an
    Apptainer container, the way to the GPU driver), and the local profile's
    tasks don't overlap."""
    import re
    import subprocess

    from napari_clemreg.clemreg.segment_flow_segmentation import _build_subprocess_env, _nextflow_config

    (tmp_path / "main.nf").write_text(_TOY_PIPELINE)
    (tmp_path / "nextflow.config").write_text(_TOY_CONFIG)
    (tmp_path / "clemreg.config").write_text(_nextflow_config("local"))
    env = {**_build_subprocess_env(), "LD_LIBRARY_PATH": "/.singularity.d/libs"}

    done = subprocess.run(["nextflow", "run", "main.nf", "-ansi-log", "false", "-profile", "local", "-c", "clemreg.config"],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)

    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.count("LD=/.singularity.d/libs") == 3, done.stdout
    spans = sorted((float(start), float(end)) for start, end in
                   re.findall(r"start=([\d.]+)\s+end=([\d.]+)", done.stdout))
    assert len(spans) == 3
    assert all(later[0] >= earlier[1] for earlier, later in zip(spans, spans[1:])), spans
