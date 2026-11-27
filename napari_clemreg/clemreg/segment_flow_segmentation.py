"""EM segmentation via Crick's AI-on-Demand (Segment-Flow), as an
alternative to the bundled empanada-dl path -- which is currently
blocked entirely on Python 3.11 (empanada-dl hard-pins numpy==1.22,
see issue #5). Shells out to `nextflow run FrancisCrickInstitute/Segment-Flow`
as a subprocess; the model itself runs in Segment-Flow's own isolated
per-model conda env, never in this process -- this process only needs
the lightweight aiod_utils/aiod_registry/pyyaml packages (no heavy ML
dependencies) to build the image manifest CSV and to generate/patch the
model config ourselves (see _build_model_config's docstring for why).

Requires Nextflow and Conda on PATH. Verified end-to-end, not just by
inspection: `segment_flow_em_segmentation` was run directly against a
small synthetic test volume, through a real Nextflow/Segment-Flow
install, producing a correctly-shaped mask back. See the inline notes
below for the real, non-obvious gotchas hit getting there -- none of
them were guessed.
"""
from __future__ import annotations

import secrets
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import tifffile

from .exceptions import ClemregError

# Segment-Flow's default branch (`master`) is not the tested/stable
# target -- aiod_napari (the reference frontend) pins to a specific
# release tag instead (see its DEFAULT_NXF_REV in
# aiod_napari/inference/nxf.py) and this must match. Confirmed directly:
# running against bare `master` HEAD failed two different ways (a Groovy
# parse error in a newer commit, then a channel-wiring bug even once that
# was worked around) that both disappeared once pinned to this tag.
SEGMENT_FLOW_REVISION = "0.2.1"

SEGMENT_FLOW_REPO = "FrancisCrickInstitute/Segment-Flow"


def _segment_flow_source() -> list[str]:
    """The pipeline argument(s) for ``nextflow run``.

    Honours AIOD_NXF_REPO the same way aiod_napari does: when set, it names
    a local Segment-Flow checkout, which is run as-is (no ``-r``: a local
    checkout is already at whatever revision it's checked out to). This
    lets a packaged install (e.g. a container) supply a pinned Segment-Flow
    with site-specific profiles, and run without fetching from GitHub.
    """
    import os

    if nxf_repo := os.environ.get("AIOD_NXF_REPO"):
        repo = Path(nxf_repo)
        if not repo.is_dir():
            raise SegmentFlowNotAvailable(
                f"AIOD_NXF_REPO was set to {repo}, which does not exist."
            )
        return [str(repo)]
    return [SEGMENT_FLOW_REPO, "-r", SEGMENT_FLOW_REVISION]


def _default_profile() -> str:
    """Nextflow profile when the caller doesn't choose one: AIOD_NXF_PROFILE
    if set (e.g. a GPU profile provided by a local AIOD_NXF_REPO checkout),
    else "local"."""
    import os

    return os.environ.get("AIOD_NXF_PROFILE") or "local"


class SegmentFlowNotAvailable(ClemregError):
    """Raised when Nextflow (or Conda) isn't available on PATH."""


class SegmentFlowRunError(ClemregError):
    """Raised when the Nextflow pipeline itself fails."""


def _build_subprocess_env() -> dict:
    """Environment for the Nextflow subprocess call.

    NXF_VER pins to a Nextflow release known to work with
    SEGMENT_FLOW_REVISION (see its comment above) -- newer Nextflow
    releases hit a real Groovy parse regression against Segment-Flow's
    pipeline script, confirmed directly. Left alone if the caller's own
    environment already sets NXF_VER.

    JAVA_HOME: Nextflow needs a JDK, and very recent ones (confirmed: 26)
    fail with "Unsupported class file major version". macOS's own
    `/usr/libexec/java_home -v <N>` isn't reliable here -- confirmed
    directly that it happily returns a *newer* JDK than requested instead
    of failing, so it can't tell us whether a compatible one is actually
    available. Instead, look directly for a Homebrew-installed
    openjdk@17/21 (the two versions the prerequisite-check error message
    below recommends), under both the Apple Silicon and Intel Homebrew
    prefixes. Left alone if the caller already set JAVA_HOME, or if no
    such install is found -- guessing further would risk pointing at a
    path that doesn't exist.

    VIRTUAL_ENV / venv's own bin/ on PATH: this is what actually broke
    every real setupModel run (previously misdiagnosed as a concurrency
    race -- serializing the pipeline via executor.queueSize didn't fix
    it, which is what led to finding this). `uv run` (and a plain venv
    activation) prepends the venv's bin/ to PATH and sets VIRTUAL_ENV,
    *without* deactivating an already-active conda env. Confirmed
    directly, step by step: with both active at once (VIRTUAL_ENV set,
    CONDA_SHLVL=1), sourcing conda's own activation script for a
    completely different, unrelated env runs successfully and changes
    nothing about the error -- `which python` before and after is
    identical, still the venv's own python -- because conda's activation
    script has no knowledge of the venv's bin/ entry and never removes
    it, so it keeps winning over whatever conda env conda just
    "activated". Nextflow's own per-process conda activation (used by
    setupModel to reach its isolated aiod_registry install) hits exactly
    this. Strip VIRTUAL_ENV and its bin/ from PATH so Nextflow's child
    processes get a clean PATH to manage their own conda environments,
    unaffected by whatever Python environment launched napari itself.
    """
    import os
    import platform

    env = os.environ.copy()
    env.setdefault("NXF_VER", "25.04.7")

    virtual_env = env.pop("VIRTUAL_ENV", None)
    if virtual_env:
        venv_bin = str(Path(virtual_env) / "bin")
        env["PATH"] = os.pathsep.join(
            p for p in env.get("PATH", "").split(os.pathsep) if p != venv_bin
        )

    if platform.system() == "Darwin" and "JAVA_HOME" not in os.environ:
        for homebrew_prefix in ("/opt/homebrew/opt", "/usr/local/opt"):
            for jdk_version in ("21", "17"):
                candidate = Path(homebrew_prefix) / f"openjdk@{jdk_version}"
                if candidate.is_dir():
                    env["JAVA_HOME"] = str(candidate)
                    break
            else:
                continue
            break

    return env


def _run_nextflow(cmd: list, cwd: Path, env: dict) -> tuple[int, str]:
    """Run the Nextflow subprocess, printing its output line-by-line as it
    arrives instead of swallowing it until the process exits.

    Nextflow's own output is what previously gave visible progress in the
    terminal for the (now-optional) empanada-dl path's tqdm bars; a plain
    `subprocess.run(..., capture_output=True)` here produced zero terminal
    feedback for the entire run (confirmed directly: nothing printed at all
    while a real run was in progress) -- unhelpful for a pipeline that can
    run for minutes, especially on a first call against a given root_dir
    while a model environment/checkpoint downloads. Still returns the full
    output so a failure can be reported with complete context.
    """
    process = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    lines = []
    for line in process.stdout:
        # flush=True: stdout is only line-buffered by default when
        # attached to a real terminal -- when napari's own output is
        # redirected to a log file (or, as found while testing this,
        # to any non-TTY pipe), Python falls back to block buffering,
        # which would silently defeat the whole point of streaming here.
        print(line, end="", flush=True)
        lines.append(line)
    process.wait()
    return process.returncode, "".join(lines)


def _check_prerequisites() -> None:
    missing = [tool for tool in ("nextflow", "conda") if shutil.which(tool) is None]
    if missing:
        raise SegmentFlowNotAvailable(
            f"Segment-Flow EM segmentation needs {' and '.join(missing)} on PATH. "
            "See https://www.nextflow.io/ and https://docs.conda.io/ to install. "
            "Note: Nextflow also needs a JDK -- very recent JDKs (tested: 26) can "
            "fail with 'Unsupported class file major version'; a JDK around 17-21 "
            "is the safer bet (e.g. `brew install openjdk@21` on macOS -- this is "
            "auto-detected and used even without setting JAVA_HOME yourself; on "
            "other platforms, set JAVA_HOME to point at a 17-21 JDK if your "
            "default is newer)."
        )
    try:
        import aiod_utils  # noqa: F401
    except ImportError as e:
        raise SegmentFlowNotAvailable(
            "Segment-Flow EM segmentation needs the `aiod_utils` package "
            "(`pip install aiod_utils`) to build the image manifest CSV in the "
            "format Segment-Flow expects."
        ) from e
    try:
        import aiod_registry  # noqa: F401
        import yaml  # noqa: F401
    except ImportError as e:
        raise SegmentFlowNotAvailable(
            "Segment-Flow EM segmentation needs the `aiod_registry` and `pyyaml` "
            "packages (`pip install aiod_registry pyyaml`) to generate and patch "
            "the model config ourselves (see _build_model_config's docstring)."
        ) from e


def _build_model_config(model_type: str, task: str, conf_threshold: float, dest_dir: Path) -> Path:
    """Generate Segment-Flow's own default config for this model/task via
    aiod_registry (the exact same call setupModel itself makes, so the
    schema is guaranteed correct -- confirmed directly, not guessed), then
    patch conf_threshold and write it out for --model_config.

    Why this exists: aiod_registry's own manifest default for MitoNet v1's
    conf_threshold is 0.5, notably stricter than empanada-dl's own bundled
    config (0.3, see empanada_configs/MitoNet_V1.yaml) -- confirmed
    directly by generating and reading Segment-Flow's actual default
    config. Everything else lines up (same checkpoint URL, same
    normalization mean/std). A real production run against a real EM
    volume (known, via the bundled empanada-dl path, to contain
    mitochondria) found none at all via Segment-Flow -- plausibly, though
    not certainly, explained by this stricter threshold.

    Segment-Flow's --model_config option (routed through setupModel's
    --user-config) *replaces* the whole config rather than overriding one
    field (confirmed directly by reading setup_model.py) -- so this
    generates the full default first and only then patches the one field,
    rather than hand-writing a config from scratch and risking an
    incomplete/wrong schema.
    """
    from aiod_registry import load_manifests
    from aiod_registry.utils import generate_default_config
    import yaml

    manifests = load_manifests()
    default_yaml = generate_default_config(manifests["empanada"], model_type, task)
    config = yaml.safe_load(default_yaml)
    config["conf_threshold"] = conf_threshold

    config_path = dest_dir / "model_config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def segment_flow_em_segmentation(
    volume: np.ndarray,
    model_type: str = "MitoNet v1",
    task: str = "mito",
    profile: str | None = None,
    root_dir: str | Path | None = None,
    conf_threshold: float = 0.3,
) -> np.ndarray:
    """Run EM segmentation via Crick's AI-on-Demand Segment-Flow pipeline.

    Parameters
    ----------
    volume : np.ndarray
        EM image volume to segment.
    model_type : str
        Segment-Flow / AIoD registry model version name (e.g. "MitoNet v1"
        -- see the "versions" keys in aiod_registry's empanada.json
        manifest for the full list, not the model's own casual name).
    task : str
        AIoD registry task key for the chosen model_type (e.g. "mito" --
        again, taken from the registry manifest, not guessed; this does
        NOT match empanada_segmentation.py's own axis_prediction concept).
    profile : str, optional
        Nextflow execution profile: "local", "crick", or "rosalind" (or
        any profile a local AIOD_NXF_REPO checkout defines). Defaults to
        the AIOD_NXF_PROFILE environment variable, else "local".
    root_dir : str or Path, optional
        AIoD cache/output root (holds conda envs, model checkpoints, and
        results -- reused across calls so models aren't re-downloaded).
        Defaults to ``~/.nextflow/aiod``, matching aiod_napari's own
        default, so a cache built via aiod_napari or a previous call is
        reused rather than duplicated.
    conf_threshold : float
        Segmentation confidence threshold passed to the model. Defaults
        to 0.3 to match empanada-dl's own bundled config (see
        _build_model_config's docstring for why this differs from
        aiod_registry's own default of 0.5).

    Returns
    -------
    np.ndarray
        The segmentation mask, same shape as ``volume``.
    """
    _check_prerequisites()
    source = _segment_flow_source()
    if profile is None:
        profile = _default_profile()

    # Match Segment-Flow's own default (nextflow.config:
    # root_dir = "${System.getProperty('user.home')}/.nextflow/aiod") so a
    # cache built by a previous call, or by aiod_napari itself, is reused
    # rather than duplicated -- and so we know where to look for the output
    # below regardless of whether the caller passed root_dir explicitly.
    if root_dir is None:
        root_dir = Path.home() / ".nextflow" / "aiod"
    root_dir = Path(root_dir)

    # aiod_utils.io.image_paths_to_csv builds the image-manifest CSV in the
    # exact format Segment-Flow expects (img_path, num_slices, height, width,
    # channels, dtype) -- a hand-written CSV with just img_path fails with
    # "Column 'height' not found" partway through the pipeline (split_stacks),
    # confirmed directly. Dimension auto-detection from file metadata isn't
    # implemented upstream (raises NotImplementedError), so dims must be
    # passed explicitly.
    from aiod_utils.io import image_paths_to_csv

    with tempfile.TemporaryDirectory(prefix="napari_clemreg_segment_flow_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        img_path = tmpdir / "input.tif"
        tifffile.imwrite(img_path, volume)

        csv_path = tmpdir / "images.csv"
        depth, height, width = volume.shape
        image_paths_to_csv(
            [img_path],
            csv_path,
            dimensions={"Z": depth, "Y": height, "X": width},
            dtypes=str(volume.dtype),
            overwrite=True,
            index=False,
        )

        model_config_path = _build_model_config(model_type, task, conf_threshold, tmpdir)
        param_hash = secrets.token_hex(4)

        cmd = [
            "nextflow", "run", *source,
            "-profile", profile,
            # No forced serialization here (see git history for an
            # executor.queueSize=1 attempt that was reverted): the real
            # cause turned out to be _build_subprocess_env() not
            # stripping a leftover VIRTUAL_ENV/venv-bin PATH entry (see
            # its docstring), not concurrent task submission. That
            # serialization "worked" in earlier testing purely because
            # those tests happened not to go through `uv run` -- it
            # never fixed the actual bug, and cost real parallelism
            # (e.g. runModel across substacks) for no reason.
            "--img_dir", str(csv_path),
            "--model", "empanada",
            "--model_type", model_type,
            "--task", task,
            "--model_config", str(model_config_path),
            "--output_format", "tiff",
            "--root_dir", str(root_dir),
            # Names this run's mask; otherwise it's a hash of all the
            # parameters, unrelated to when the mask was made.
            "--param_hash", param_hash,
        ]

        print(
            f"Running Segment-Flow EM segmentation (model_type={model_type!r}, "
            f"task={task!r})... this shells out to a real Nextflow pipeline and "
            "can take a while, especially on first run against a given "
            "root_dir while the model environment/checkpoint downloads. Note: "
            "Nextflow may print an 'ERROR ~' block for an individual failed "
            "process attempt (e.g. a transient conda-environment-setup issue) "
            "and still go on to complete the run successfully -- confirmed "
            "directly. That block alone isn't a reliable failure signal; this "
            "function only raises once Nextflow's own final exit code says the "
            "whole run failed, so let it keep running rather than assuming "
            "failure from an ERROR block partway through.",
            flush=True,
        )
        # A previous version of this code retried on a "setupModel +
        # ModuleNotFoundError" signature, believing it was a transient
        # race between concurrently-submitted Nextflow tasks. Confirmed
        # directly that diagnosis was wrong: the real cause (see
        # _build_subprocess_env()'s docstring -- a leftover venv bin/ on
        # PATH shadowing the conda env Nextflow activates) is
        # deterministic, not transient -- retrying 3 times against it
        # failed identically all 3 times, live. No retry here now that
        # the actual cause is fixed at the source; a real failure should
        # surface immediately rather than being masked behind a pointless
        # wait.
        returncode, output = _run_nextflow(cmd, cwd=tmpdir, env=_build_subprocess_env())

        # .nextflow.log lands in cwd (tmpdir), which is deleted with the
        # `with` block above -- so without this, the one file that would
        # help debug exactly this kind of transient mid-run error is gone
        # before anyone can look at it. Keep it on failure, when it's
        # actually needed; skip on success to avoid piling up clutter.
        log_path = tmpdir / ".nextflow.log"
        saved_log_path = None
        if returncode != 0 and log_path.exists():
            log_dir = root_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            saved_log_path = log_dir / f"nextflow_{time.strftime('%Y%m%d_%H%M%S')}.log"
            shutil.copy(log_path, saved_log_path)

        if returncode != 0:
            log_note = f"\n\nFull Nextflow log saved to {saved_log_path}" if saved_log_path else ""
            raise SegmentFlowRunError(
                f"Segment-Flow failed (exit {returncode}):\n{output}{log_note}"
            )
        print("Segment-Flow EM segmentation finished.", flush=True)

        # combineStacks publishes the result as
        # <root_dir>/aiod_cache/<model>/<model_type>_masks/<image-id>_masks_<param_hash>_all.<format>,
        # next to every earlier run's mask, so read this run's by its hash.
        # The image id is the input's file name, input.tif.
        mask_path = root_dir / "aiod_cache" / "empanada" / f"{model_type}_masks" / f"input_tif_masks_{param_hash}_all.tiff"
        if not mask_path.exists():
            raise SegmentFlowRunError(
                f"Segment-Flow reported success but no output mask was found at "
                f"{mask_path}. Nextflow output:\n{output}"
            )
        mask = tifffile.imread(mask_path)
        # Each run's files are named by its hash, so they're never reused.
        mask_path.unlink()
        (root_dir / "aiod_cache" / "splits" / f"substacks_{param_hash}.csv").unlink(missing_ok=True)
        return mask
