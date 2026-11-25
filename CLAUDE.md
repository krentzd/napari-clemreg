# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Remotes

Work happens in the original repository, `krentzd/napari-clemreg` (remote `upstream`), on its `version-2` branch (since 2026-11-25). Push branches and open PRs there. `origin` is the fork `martlj/napari-clemreg`, where the modernisation work began; its issue tracker and earlier PRs stay there. In PRs on `krentzd/napari-clemreg`, refer to the fork's issues as `martlj/napari-clemreg#N`, since a bare `#N` means that repository's own issue.

## Branching

`version-2` on `krentzd/napari-clemreg` is the integration branch. It started from the fork's `modernisation` branch (134db6f). `main` there is the published version and stays the default branch: don't commit this work to it. Work happens on one branch per issue, e.g. `6-mobie-export`, merged into `version-2` via PR. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow before starting new work.

## Commands

The maintainer's day-to-day setup is pixi (`pixi.toml`; Python 3.11, PyQt6, Nextflow and JDK 21 from conda-forge/bioconda, napari-clemreg editable with the `segment-flow` extra; Apple-silicon macOS and Linux with glibc 2.31 or newer):

```
pixi run napari             # default environment
pixi run -e test test       # test environment: pytest, pytest-qt, pytest-cov
pixi run -e test test-slow  # also the slow tests (see below)
```

Elsewhere, any Python 3.11+ environment works:

```
pip install -e ".[segment-flow]" pyqt6
```

Code changes need only a napari restart. After editing `setup.cfg`'s dependencies, run `pixi update napari-clemreg`: plain `pixi install` doesn't notice `setup.cfg` changes (checked). Commit `pixi.lock` with `pixi.toml` changes. Segment-Flow still uses the system `conda` (Miniconda) to build model environments.

Run the test suite (uses tox, matching CI's per-Python/per-OS matrix):

```
tox
```

Run tests directly with pytest against the active environment:

```
pytest -v --color=yes --cov=napari_clemreg --cov-report=xml
```

Run a single test file or test:

```
pytest napari_clemreg/_tests/test_warp_image_volume.py
pytest napari_clemreg/_tests/test_warp_image_volume.py::test_name
```

Widget tests: `test_dock_widget.py` checks that every widget and dock widget builds. `test_widget_behaviour.py` runs them: it calls each widget and clicks each **Run this step** button against a `napari.components.ViewerModel` (no real window, so it works locally under `QT_QPA_PLATFORM=offscreen`), with the `widget_components.run_*` pipeline functions replaced by recording stubs. Add a test there for any change to widget wiring. Known open GUI bugs are pinned there as `xfail(strict=True)` tests naming their issue, so remove the marker when fixing one. Unit tests for the pure math in `clemreg/` are also under `napari_clemreg/_tests/` (see §0 of the modernisation plan below).

**macOS: torch + napari can segfault on import** (two OpenMP runtimes loaded in one process — torch's bundled `libiomp5` vs. the conda-forge `libomp` napari's stack pulls in). `napari_clemreg/_tests/conftest.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` before those imports, so pytest runs are unaffected; if you import `napari_clemreg` (or `torch` then `napari`) directly in a script or REPL, set those env vars first. No-op on Linux CI.

Two kinds of tests are opt-in:
- **Slow** (`@pytest.mark.slow`, run with `--run-slow`): real sample-data downloads, reader checks on the real Zenodo files, and `test_gui_end_to_end.py`, which runs Run Registration step by step on the real sample data, once from the Zenodo TIFFs and once streamed as OME-Zarr from the BioImage Archive (about 2–3 minutes each, peaking at about 8 GB of memory; the OME-Zarr run needs the network every time, since nothing is cached). CI's per-push `tests` workflow skips them. The `slow tests` workflow runs them after each merge into `version-2` (except docs-only changes) and on demand (Actions tab → "slow tests" → Run workflow, choosing `version-2`). Its weekly schedule only runs from the default branch, so it's inactive until this work reaches `main`. With tox, pass the flag through: `tox -- --run-slow`.
- **Real Viewer** (`test_real_viewer.py`, runs only with `CLEMREG_REAL_VIEWER=1`): opens each dock widget in a real napari Viewer through the npe2 manifest. CI sets the variable on Linux. Locally it works with the native macOS Qt platform but segfaults under `QT_QPA_PLATFORM=offscreen`, which is why it's opt-in.

There are older local conda envs for this repo. Use pixi (above) for day-to-day work; these are kept only for reference:
- `clemreg_env` (python 3.9) — matches the **pre-#3** `setup.cfg` pins, and was used for the §0 characterisation baseline. It can't run the current code: `python_requires` is `>=3.11`, and `zarr >= 3.1` needs Python 3.11.
- `clemreg-py311-verify` (python 3.11) — built to verify the #3 (Python 3.11) dependency bump, with every `setup.cfg` dependency except `empanada-dl`. It predates the `zarr`, `dask`, `fsspec[http]` and `aiohttp` dependencies, so extend it before using it.
- `napari-clemreg-311` and `napari-clemreg` (no suffix) — both broken (numpy/scikit-image ABI mismatches from earlier, less careful upgrade attempts). Not worth fixing; superseded by pixi.

**macOS: open3d's wheel needs `libusb`** (`brew install libusb`) or it fails to import with a `dlopen`/`Library not loaded` error. Not a napari-clemreg issue — open3d's compiled binary links against it unconditionally.

## Architecture

The plugin has two layers under `napari_clemreg/`:

- **`widgets/`** — thin `magicgui`-based napari GUI entry points (`run_registration.py`, `fixed_segmentation.py`, `moving_segmentation.py`, `point_cloud_sampling.py`, `registration_warping.py`). These are registered as napari commands/widgets in `napari_clemreg/napari.yaml`, the plugin manifest read by napari's plugin engine (npe2). Widget functions handle UI wiring, parameter forms (`clemreg/on_init_specs.py` defines widget layout specs), and threading via napari's `thread_worker`/`GeneratorWorker`.
- **`clemreg/`** — the actual algorithm/glue code, napari-independent aside from layer types and threading helpers:
  - `widget_components.py` — glue functions (e.g. `run_moving_segmentation`) called by the widgets, chaining together the pipeline steps below.
  - `log_segmentation.py`, `empanada_segmentation.py` — FM (Laplacian-of-Gaussian) and EM (MitoNet/empanada-dl) segmentation.
  - `mask_roi.py`, `data_preprocessing.py` — ROI masking and isotropic resampling.
  - `point_cloud_sampling.py`, `point_cloud_registration.py` — point-cloud extraction from segmentations and registration (Rigid/Affine CPD, BCPD via `probreg`).
  - `warp_image_volume.py` — warps the moving image volume using the resulting point-cloud transform.
  - `sample_data.py` — provides the built-in napari sample dataset.
- `napari_clemreg/_reader.py` — napari file reader hook (`.tif`/`.tiff`, directories).

The full pipeline (EM segmentation → FM segmentation → point-cloud sampling → point-cloud registration → image warping) can be run end-to-end via the `Run registration` widget, or each stage run independently via the numbered "split registration" widgets — both paths call into the same `clemreg/` functions.

### In-progress package split

`packages/clemreg/` is a placeholder (version `0.0.0`, reserving the PyPI name) for a standalone, napari-free `clemreg` core package, which the plugin will depend on. **Read [docs/design/package-split.md](docs/design/package-split.md) before changing anything in `napari_clemreg/clemreg/`**. It defines the boundary: the four `run_*` functions in `widget_components` stay as the plugin's layer-to-array adapters, and the core takes arrays, `PixelSize` (µm), `Params` and `Transform`, and raises `ClemregError` subclasses. Status is on [#7](https://github.com/martlj/napari-clemreg/issues/7) and its phase sub-issues (#57–#61), not in the doc. Record design changes in the doc's decisions log, not as "update" notes.

The original modernisation plan (testing foundation, Python 3.11, napari upgrade, MoBIE export, and the repo/PyPI/release strategy; its §6 on the split now points to the design doc) is in [docs/napari-clemreg-modernisation-plan.md](docs/napari-clemreg-modernisation-plan.md) — read it before starting work related to any of those areas.

## Shared napari skill

`.claude/skills/napari-dev/` holds napari's plugin guidelines as a Claude Code skill. The maintainer's master copy is a personal skill outside this repository; this committed copy is for collaborators, so keep it general and put anything specific to this repo in this file instead. When the guidelines change, copy the new version in on a branch.

## Packaging

The root package (`napari-clemreg`, the napari plugin) is defined via `setup.cfg`/`setup.py` (no `pyproject.toml`). `python_requires = >=3.11` as of the #3 modernisation work, with most pins loosened to `>=` per the plan's §1.2 table. Heavy dependencies (`napari`, `open3d`, `probreg`) make installs slow.

`empanada-dl` (needed only by the bundled EM Segmentation/MitoNet widget, along with `torch`) is **not** in `install_requires` — it lives in its own `empanada` extra (`pip install napari-clemreg[empanada]`). It's still hard-pinned at `==0.1.7`, and its own `numpy==1.22` pin still has no Python 3.11 wheels, so that one extra still won't resolve on Python 3.11 (see [issue #5](https://github.com/martlj/napari-clemreg/issues/5)) — but this is now scoped to just that extra rather than blocking the base install entirely, since a normal `pip install napari-clemreg` (or the `segment-flow` extra, see below) resolves and works fine without it (verified empirically, both in `clemreg-py311-verify` and via a clean dependency-resolution dry run against Python 3.11).

For EM segmentation on Python 3.11 today, use the `segment-flow` extra instead (`clemreg/segment_flow_segmentation.py`) — see the widgets' `EM Segmentation Backend` dropdown.

## CHANGELOG and ROADMAP

Keep [CHANGELOG.md](CHANGELOG.md) and [ROADMAP.md](ROADMAP.md) in step with every merged PR (standing instruction, 2026-09-23).

**In each PR you open:** update both files in the PR itself, so they're correct the moment it merges.
- CHANGELOG: add an entry under `[Unreleased]`, linking the PR and issue. Use `Added`/`Changed`/`Fixed`/`Removed` for anything users would notice. Tests, CI and project docs go under `Development`. Mark breaking changes `**Breaking:**`, and check that the `[Unreleased]` header's proposed version still fits the bump rules below.
- ROADMAP: remove or rewrite anything the PR finishes, and add follow-ups, open decisions or bugs it uncovers, with issue links. The ROADMAP only lists what's still to do. What's done belongs in the CHANGELOG.
- If the PR closes an issue, or finishes a sub-issue of #7, check that the issue (and the [package split design](docs/design/package-split.md), if relevant) agrees.

**After a merge:** when you're told a PR has merged, and at the start of a session before other work, cross-check. List the PRs merged into `version-2` since the newest PR linked in the CHANGELOG (`gh pr list -R krentzd/napari-clemreg --base version-2 --state merged`). Also look for ROADMAP items those PRs finished and issues they fixed that are still open. Fix any gaps in one catch-up PR. Don't merge it, as with any other PR.

## Closing issues

PRs into `version-2` don't close issues on merge, because it isn't the default branch: close them by hand (step 2 below). Closing keywords still matter, because they fire once `version-2` merges into `main`, and they did on the fork, where `modernisation` was the default. GitHub closes an issue when a PR merged into the default branch (or a commit message) contains a closing keyword before its number: *close(s/d)*, *fix(es/ed)*, *resolve(s/d)*. It matches the keyword **anywhere**, including prose, negations and future tense. "doesn't fully fix #53" closed #53 (PR #67), and "(fixes #47)", written about planned work, closed #47 while it was still broken (PR #62; reopened).

- **Only** write a closing keyword before an issue number when the PR completely resolves that issue, e.g. `Fixes #50` in the summary.
- Everywhere else, refer to issues without one: "part of #58", "for #47", "see #53", "#47 is planned for phase 1". Avoid "fix"/"close"/"resolve" directly before `#N` in any wording, and check the PR description for this before opening it.

**After a merge** (as part of the cross-check under *CHANGELOG and ROADMAP*):
1. Check every issue the merge closed was actually resolved by it. Reopen any closed by mistake, with a comment explaining why.
2. Close issues the merge **unambiguously** completed, with a comment linking the PR. That means:
   - every task-list item is ticked, and the work for them is merged; or
   - the issue's stated outcome is demonstrably delivered, e.g. the strict-xfail test pinned to it now passes and its marker has been removed; or
   - it's a parent issue (like #7) and all its sub-issues are closed.
3. Tick task-list items the merge completed, on the issue itself and on tracking issues such as #8 and #58.
4. Don't close anything ambiguous, e.g. partly done, or needing a decision or a manual check. List those for the user instead.

## Issue labels for what's waiting

Two labels mark issues that can't move forward through ordinary work. Each labelled issue has a comment saying exactly what it's waiting on.

- **`needs-decision`**: waiting on a decision from the maintainer, e.g. what output a feature should produce, or an open decision in the design doc. [All issues with it](https://github.com/martlj/napari-clemreg/issues?q=is%3Aopen+label%3Aneeds-decision).
- **`blocked`**: waiting on something outside this repo, such as a person (e.g. the upstream owner) or an upstream project (e.g. `empanada-dl`). An issue that's just waiting for the previous phase or sub-issue isn't `blocked`; that order is already shown by the sub-issues. [All issues with it](https://github.com/martlj/napari-clemreg/issues?q=is%3Aopen+label%3Ablocked).

Keep them current as part of the post-merge check, and whenever a decision is made or a blocker clears: remove the label, with a comment recording the decision or what unblocked it, and update the design doc's decisions log if relevant. Add the label, with its comment, when new work turns out to need a decision or depends on someone else. Decisions with no issue of their own (e.g. D7 and D8 in the design doc) stay in the ROADMAP's *Open decisions* section.

## Versioning

`setup.cfg`'s `version` (currently `0.2.1`) hasn't been bumped since this fork's modernisation work began, despite several release-worthy batches of work having landed. See [CHANGELOG.md](CHANGELOG.md)'s status note and [issue #1](https://github.com/martlj/napari-clemreg/issues/1) for the real release/versioning decision, which hasn't been made yet.

When a merged (or about-to-be-merged) change is release-worthy, suggest a follow-up commit that bumps `setup.cfg`'s `version` and adds the matching `CHANGELOG.md` entry. Don't apply it unilaterally: versioning and release timing are the user's call. Every suggestion must say **which level (patch, minor or major) and why**, using the rules below. Don't default to minor.

`setup.cfg`'s `version` is the only place the version is defined (`napari_clemreg/__init__.py` reads it via `importlib.metadata`). It changes only in a release commit. That commit sets `version`, renames the CHANGELOG's `[Unreleased]` heading to that version with the date, and is the commit that gets tagged (`vX.Y.Z`). Between releases, `setup.cfg` keeps the last released version. Never go back and bump it for CHANGELOG batches that were never tagged or published: those numbers are just labels, and the first real release sets `version` straight to the latest one. Tags and releases belong on the canonical upstream or Crick-org repo, not this fork (plan §7.7).

### Choosing the bump level

The "public surface" here means: the widgets and their parameters, the `napari.yaml` command/widget/sample-data IDs, importable Python APIs, the install requirements and extras (Python version, `install_requires`, extra names), and the outputs users get by default (layer shapes, scales, file formats).

- **Patch** (`x.y.Z`): bug fixes only, with no new features and no change to the public surface. A fix that restores the documented or intended behaviour is a patch, even if it changes results.
- **Minor** (`x.Y.0`): new features, new options or new backends added in a backwards-compatible way, deprecations, and changed defaults that still leave the old behaviour available as an option.
- **Major** (`X.0.0`): anything that breaks existing users, such as removing or renaming a widget, parameter, extra or import path, raising the minimum Python, adding a hard dependency that won't install somewhere it used to, or changing default outputs with no way back to the old ones.
- **No bump:** docs, CI, tests or refactors that ship nothing different to users. These go in the next release's entry, if they're worth mentioning at all.

A batch takes the highest level of any change in it. A fix to a feature added in the same unreleased batch is part of that feature, not a separate patch.

**While pre-1.0** (the current state; plan §7.7): anything that would be a major bump is a **minor** bump instead. Flag it as `**Breaking:**` in the CHANGELOG entry, and still bump patch for fix-only batches. SemVer's 0.x rules allow anything to change, but keeping the patch/minor distinction tells users which upgrades are safe.

### When to declare 1.0.0

1.0.0 is a promise that the public surface is stable. It doesn't measure how much has changed. Suggest it only once all of these are true, and say which of them still aren't:

1. The package split ([#7](https://github.com/martlj/napari-clemreg/issues/7)) has landed and settled, and the public API of `clemreg` (the core package) and of `napari_clemreg` is written down (what's public and what's internal).
2. No known breaking changes are still queued on [ROADMAP.md](ROADMAP.md). For example, the OME-Zarr work must either have landed or be designed so it only adds to the current behaviour.
3. Every documented install path resolves on the supported Pythons: the `empanada` extra is fixed or dropped ([#5](https://github.com/martlj/napari-clemreg/issues/5)).
4. Widget-level tests exist, so the stability promise can actually be checked in CI.
5. The release is cut from the canonical upstream or Crick-org repo, not this fork ([#1](https://github.com/martlj/napari-clemreg/issues/1), plan §7.7), so the tag gets the Zenodo DOI and `CITATION.cff`.

After the split, `clemreg` and `napari-clemreg` are versioned independently. Each reaches 1.0 on its own merits, with criterion 1 applied to that package's own API. The plugin is tagged `vX.Y.Z` and the core `clemreg-vX.Y.Z`, and each has its own CHANGELOG (`CHANGELOG.md` and `packages/clemreg/CHANGELOG.md`). See *Two packages in one repository* in the [package split design](docs/design/package-split.md).
