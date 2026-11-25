# Design: splitting `clemreg` (core) from `napari-clemreg` (plugin)

This document describes the design and the reasons for it. **It doesn't track status**: that's on GitHub, in [#7](https://github.com/martlj/napari-clemreg/issues/7) (the parent issue), its sub-issues (one per phase below) and the [*clemreg 0.1.0* milestone](https://github.com/martlj/napari-clemreg/milestone/1). When a decision changes, update the [decisions log](#decisions-log) rather than adding "update" notes to the text.

It replaces §6 of the [modernisation plan](../napari-clemreg-modernisation-plan.md#6-splitting-into-two-packages-clemreg-core--napari-clemreg-widget), which was written on 2026-09-21. Since then the code gained Segment-Flow, the per-step buttons (#35), native-resolution warping (#33), bioio metadata (#37) and widget behaviour tests (#54).

## Goal

Two installable distributions from one repository:

- **`clemreg`** (import `clemreg`): the pipeline. FM and EM segmentation, point cloud sampling, registration, warping and, later, MoBIE export. It works on numpy arrays and plain Python types, with **no napari, Qt or magicgui imports**. This makes headless and batch use possible (`notebooks/clemreg_batch_mode.ipynb` already does this, but has to import napari to do it), as well as scripting and fast tests without Qt.
- **`napari-clemreg`** (import `napari_clemreg`): the napari plugin. Widgets, the npe2 manifest, the reader, sample data, and thin adapters that turn layers into arrays, call `clemreg`, and turn the results back into layers. It depends on `clemreg`.

## Where the code is now

The pipeline code lives in `napari_clemreg/clemreg/`. Most modules use napari only lightly: they read `.data` or `.metadata`, or build a layer to return.

| Module | napari / Qt use today | Goes to |
|---|---|---|
| `log_segmentation.py` | takes `Image`; imports `thread_worker` (unused) | core, arrays in and out |
| `mask_roi.py` | takes `Shapes`, returns `Image` | core: takes polygon vertices and a z range |
| `point_cloud_sampling.py` | takes `Labels`, returns `Points` | core: returns an `(N, 3)` array |
| `point_cloud_registration.py` | `PointsData` (an ndarray alias), `magicgui.tqdm`, a viewer-based `_add_data` (unused) | core: plain `tqdm` or a progress callback; drop `_add_data` |
| `warp_image_volume.py` | public functions take and return `Image`/`Points` and read `.colormap`/`.blending`; the private maths is pure numpy | core: array warps; layer styling and linked channels go to the plugin |
| `data_preprocessing.py` | `make_isotropic` and `return_isotropic_image_list` take `Image`; the rest works on arrays and dicts | core, arrays only |
| `empanada_segmentation.py` | none | core, `[empanada]` extra |
| `segment_flow_segmentation.py` | none | core, `[segment-flow]` extra |
| `widget_components.py` | the four `run_*` functions both GUIs call; `show_info`; `'No segmentation'` string returns; broken JSON helpers (#47) | **split**: pipeline steps go to the core; the `run_*` functions stay in the plugin as adapters (see below) |
| `sample_data.py` | none; the pooch downloads and metadata return napari layer tuples | core as `clemreg.data` (arrays plus `PixelSize`); the plugin wraps them as layer tuples for `napari.yaml` (D5) |
| `_reader.py` | none (bioio) | napari reader hook in the plugin; pixel-size reading in the core as `clemreg.io` |
| `on_init_specs.py`, `_qt_layout.py`, `_napari_compat.py`, `widgets/*`, `napari.yaml` | UI | plugin |

Unused legacy functions (`warp_image_volume_deprecated`, `_make_isotropic_v1`, `filter_binary_segmentation_v1`, and probably `point_cloud_registration._add_data`) are deleted in phase 1 rather than moved.

## The boundary

### Keep the four `run_*` functions as the plugin's adapters

Since #35, both GUIs (the split widgets and Run Registration's **Run this step** buttons) go through the same four functions in `widget_components`: `run_fixed_segmentation`, `run_moving_segmentation`, `run_point_cloud_sampling` and `run_point_cloud_registration_and_warping`. `test_widget_behaviour.py` replaces exactly these with stubs.

So these four keep their names and their layer-based signatures, and become the plugin's adapters (moving to `napari_clemreg/_adapters.py`, with `widget_components` re-exporting them). Each one turns layers into arrays and a `PixelSize`, calls one core function, and wraps the result as a layer. The widget tests keep working unchanged and become the acceptance test for the boundary.

### Core API (napari-free)

```python
from clemreg import PixelSize, Params, Result

segment_fm(volume, sigma, threshold, size_filter=None, roi=None) -> np.ndarray      # labels
segment_em(volume, backend="segment-flow") -> np.ndarray                           # labels
sample_point_cloud(labels, pixel_size, every_k, voxel_size, sigma) -> np.ndarray    # (N, 3)
register_point_clouds(moving, fixed, algorithm, max_iterations,
                      progress=None) -> Transform
warp_volume(volume, transform, moving_pixel_size, target_pixel_size,
            order, approximate_grid, sub_division_factor) -> tuple[np.ndarray, PixelSize]
run_clemreg(fm_channels, fm_pixel_size, em, em_pixel_size, params) -> Result
```

What's different from the 2026-09-21 plan:

- **`warp_volume` takes a target pixel size, not an output shape, and returns the pixel size of its output.** Native-resolution warping (#33) produces an array that's placed over EM using its scale, so the size is part of the result. The plugin turns it into the layer's `scale`.
- **The registration direction** (FM → EM or EM → FM) is a field of `Params`, handled in `run_clemreg`, rather than something only the widget knows about.
- **`Transform`** holds either an affine matrix (Rigid/Affine CPD) or thin-plate-spline control points (BCPD), so that `warp_volume` and, later, the MoBIE exporter can use it without knowing which algorithm produced it.
- **Several channels are passed as a list.** `get_linked_layers` is a napari concept. The plugin collects the linked channels and passes a list, and the core warps each one with the same transform.

### Types

- **`PixelSize(z, y, x)`**: a frozen dataclass, in **micrometres**. bioio already reports µm. Parsing `pint` strings stays in the plugin, which converts the widget's `QuantityEdit` values at the boundary (D2). Moving from today's nm to µm doesn't change results: the pipeline only uses *ratios* of pixel sizes when resampling. **Voxel Size** is measured in pixels of the resampled working grid, not in physical units, so its meaning, and saved parameter files, are unaffected.
- **`Params`**: a dataclass of every pipeline parameter (the fields of the widget sections, with the same defaults as `on_init_specs.py`), with `to_json()` / `from_json()`. Run Registration's **Save parameters** and **Parameters from JSON** use it, which fixes #47. The field names are the JSON keys, and `from_json` ignores unknown keys, so old files still load.
- **`MaskRoi(polygon, z_min, z_max)`**: the FM Mask ROI as (y, x) polygon vertices and a z range, replacing the napari `Shapes` layer. The plugin builds it from the `Shapes` layer.
- **`Result`**: the segmentations, point clouds, `Transform`, and warped channels with their pixel size. It's what the headless path returns and what MoBIE export (#6) will take as input.

### Errors

The core raises its own exceptions, all subclasses of `ClemregError(Exception)`:

- `NoSegmentationError` replaces the `'No segmentation'` string that the core returns today. That string was the cause of #25 and #50.
- `BackendNotInstalledError` says an EM segmentation backend's optional packages are missing (`missing` lists them), before anything is imported. The plugin's adapter turns it into a message about the widget's dropdown.
- `SegmentFlowNotAvailable` and `SegmentFlowRunError` are re-based onto `ClemregError`. At the moment they subclass `RuntimeError`, which superqt's generator workers treat as a deleted worker and swallow, causing the Register hang in #53. The plugin's adapters must also stop any other `RuntimeError` reaching a generator worker (#53).

The core never shows GUI messages. `show_info` calls become `logging` calls, and progress is reported through an optional `progress` callback, which the plugin connects to napari's progress bars.

### Keeping the boundary in place

A test imports every core module in a subprocess with napari, magicgui, qtpy and superqt blocked (`sys.modules[name] = None`), and fails if any of them imports. It's added in phase 1, before any files move, so the boundary is checked from then on and nobody has to rely on review to catch a stray import.

## Layout and packaging

```
napari-clemreg/                  # repo; the root package stays the plugin
  pyproject.toml                 # napari-clemreg (replaces setup.cfg/setup.py); uv workspace root
  napari_clemreg/                # plugin code, unchanged location
  packages/clemreg/
    pyproject.toml               # clemreg (hatchling)
    src/clemreg/
```

The plugin stays at the repo root, so paths in CI, tox, `napari.yaml` package data and the docs don't change, and `git blame` history is kept. The root `pyproject.toml` declares a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) with `packages/clemreg` as a member, so a single `uv pip install -e ".[segment-flow]"` gives an editable install of both packages (D1).

Only uv knows to take `clemreg` from the workspace. Until `clemreg 0.1.0` is on PyPI (only the `0.0.0` placeholder is there now), the plugin's `clemreg>=0.1,<0.2` requirement can't be met from PyPI. So a plain `pip install -e .` fails to resolve during development, and so does tox, which uses pip. Phase 2 therefore has to:

- document the pip route as `pip install -e packages/clemreg -e ".[segment-flow]"` (core first);
- switch tox to [`tox-uv`](https://github.com/tox-dev/tox-uv), or install `packages/clemreg` explicitly in tox's `deps`;
- and, when releasing (phase 4), publish `clemreg` before `napari-clemreg`.

**Core dependencies:** numpy, scipy, scikit-image, open3d, probreg, transforms3d, connected-components-3d, tqdm.
**Core extras:** `[segment-flow]` (aiod_utils), `[empanada]` (empanada-dl, torch; still blocked on Python 3.11, #5), `[io]` (bioio, bioio-tifffile), and later `[mobie]`.
**Plugin dependencies:** `clemreg>=0.1,<0.2`, `napari>=0.6`, magicgui, superqt, pint, pooch, and `clemreg[io]`. The plugin's `[segment-flow]` and `[empanada]` extras just pass through to the core's.

**Tests:** the pure tests (maths, the synthetic end-to-end registration) move to `packages/clemreg/tests`, and a separate CI job runs them without Qt or xvfb. The widget, real-Viewer and GUI end-to-end tests stay with the plugin.

**Backwards compatibility:** `napari_clemreg.clemreg.<module>` re-exports from `clemreg` and raises a `DeprecationWarning`, so anyone importing the internals keeps working. The people most likely to do that are running copies of the batch notebook published with the paper, and they won't upgrade in step with releases. So the re-exports stay **until 1.0.0, or for at least six months after the split is released, whichever is later** (D4).

### Two packages in one repository

Each package has its own release stream, so they're kept apart in the repository too. This also means the core can later move to a repository of its own (D1 option c) without untangling anything.

- **Tags:** the plugin keeps `vX.Y.Z` (as for the existing `v0.2.1`), and the core is tagged `clemreg-vX.Y.Z`. Each release workflow runs only on its own tag pattern.
- **Changelogs:** the core gets its own `packages/clemreg/CHANGELOG.md` in phase 2. The root `CHANGELOG.md` stays the plugin's, and says when a plugin release raises its `clemreg` requirement.
- **Self-contained core:** everything the core needs to build, test and release is under `packages/clemreg`: its `pyproject.toml`, README, a copy of the LICENSE, its tests and its CHANGELOG. Nothing in it refers to files elsewhere in the repo. The core's CI job runs only on changes under `packages/clemreg/`. The plugin's jobs run on every change, because the plugin depends on the core.
- **Moving the core out later** is then a `git filter-repo` of `packages/clemreg` that keeps its history. To keep the history from before phase 2 as well, the filter also has to keep `napari_clemreg/clemreg/` (the core's location until then). Open issues for the core would be transferred with it.

## Phases

Each phase is one PR (or a few), leaves the tests passing, and has its own sub-issue under #7.

0. **Confirm the decisions** below (#57). **Before phase 2 moves any files**, get the upstream owner's agreement to the new layout and import paths (D1, D4): this work is meant to go upstream as a PR (plan §7). Phase 1 is a refactor inside the current layout, so it doesn't need to wait.
1. **Remove napari from the core modules, without moving any files** (#58). Arrays in and out, layer handling moved into the adapters, plus `PixelSize`, `Params`, `Transform`, `Result` and the exceptions, and the sample-data loaders as napari-free functions (D5). Add the import-boundary test and delete the legacy functions. The widget behaviour tests are the acceptance test. This also fixes #47, #50 and #53. Most of the risk is in this phase.
2. **Move the code into `packages/clemreg`** (#59). Add both `pyproject.toml` files, the uv workspace, the pip and tox install routes described under *Layout and packaging*, and the compatibility re-exports, and give the core its own CI job that doesn't need Qt. Start the core's own CHANGELOG, and keep `packages/clemreg` self-contained (see *Two packages in one repository*). Needs the upstream owner's agreement first (see phase 0).
3. **Headless path** (#60): `run_clemreg`, with the batch notebook switched to `import clemreg` and `clemreg.data` for its sample data (no napari import), and the README and CLAUDE.md updated for the two-package install.
4. **Release** (#61): `clemreg 0.1.0` (its first real release after the `0.0.0` placeholder), and the next minor `napari-clemreg`, marked **Breaking** because internal import paths change (see CLAUDE.md § Versioning). Publish `clemreg` first, because the plugin requires it. Tag the core `clemreg-v0.1.0`, and the plugin `vX.Y.Z` as before.

The MoBIE exporter (#6) is written directly against `Result` and `Transform` in the core. It can **start as soon as phase 1 merges**, since that's when those types exist, and doesn't need to wait for the packaging work or the release. It ships as an additive `clemreg 0.2.0`. The old plan built it before the split to avoid moving it later. Now that there are widget tests, splitting first is simpler, and it keeps MoBIE from holding up the core's first release.

## Risks

- **Linked channels:** if the plugin doesn't collect every linked layer before calling the core, a multi-channel warp silently drops channels. Linked layers also come back from `get_linked_layers` as a `set`, so the order of the warped channels isn't fixed today. The adapter should pass the core an ordered list (the selected layer first, then the others by name). The output baseline already covers two linked channels, comparing them by name.
- **Numerical drift from moving code:** phase 1 changes signatures, not maths. `test_pipeline_baseline.py` runs the pipeline through the four adapters on synthetic data (FM segmentation with and without the size filter and a Mask ROI; point cloud sampling at anisotropic pixel sizes; Rigid CPD, Affine CPD and BCPD; the legacy EM-grid output; two linked channels) and compares every output with saved values at tight tolerances. The pipeline is deterministic, and the baseline catches a 0.05-pixel shift in the registration matrix. Every phase 1 PR must pass it unchanged. Regenerate it only for a change that's meant to alter results, and say why in the PR. The synthetic end-to-end test and the slow GUI test on real data must also still pass.
- **The registration result lives in a napari layer.** For Rigid and Affine CPD, the "transformed" point cloud is the moving points unchanged, with the registration matrix attached as the layer's `affine`, and the warp step reads the matrix back from `transformed.affine.affine_matrix`. So today there's no representation of the result outside napari. `Transform` replaces this, and it's the deepest napari coupling phase 1 has to undo. The adapter can keep returning a `Points` layer with that `affine` for the GUI.
- **EM → FM direction is broken** (#68): it crashes for every algorithm, and has since before the fork. Phase 1 should keep that behaviour (the baseline pins it as a strict xfail) rather than change it by accident. Fixing it needs its own decision about what the output should be.
- **Units:** `get_pixelsize` currently returns values and a unit from ImageJ-style metadata, while bioio returns µm. Converting everything to µm at the boundary must not change the pixel sizes used by the pipeline. Pin them with a test on the sample data's metadata before changing anything.
- **Two version streams:** the plugin's `clemreg>=0.1,<0.2` pin has to move along with the core's minor releases. Separate tags and changelogs keep the two release histories apart (see *Two packages in one repository*).

## Decisions log

Each decision is **Proposed** until confirmed in the phase 0 sub-issue (#57). D1–D6 were confirmed on 2026-09-23, with D1 and D4 amended and D5 reversed from what was first proposed.

| # | Decision | Options considered | Chosen | Status |
|---|---|---|---|---|
| D1 | Repository layout | (a) both packages under `packages/*/src`; (b) plugin stays at the root and the core goes in `packages/clemreg`, as a uv workspace; (c) two repositories | (b): least disruption to paths, CI and history. Amended: pip and tox can't take `clemreg` from the workspace, so phase 2 documents a core-first pip install and moves tox to `tox-uv` (or installs the core explicitly), and phase 4 publishes the core first. Needs the upstream owner's agreement before phase 2. Revisited on 2026-10-09 and (b) kept: the core's API is still changing in phase 1, the output baseline runs through the plugin's adapters, two repos wouldn't make development installs simpler, and the upstream owner would still have to agree, because the plugin's imports change either way. (b) also keeps (c) cheap (see *Two packages in one repository*). Move to (c) if any of these happens: the upstream owner declines this layout or the transfer (#1), in which case the core starts in the Crick org; projects other than the plugin depend on the core (for example the samplers, #75, or MoBIE export); the core has several releases in a row with no plugin changes; or the two packages get different maintainers. | Accepted, 2026-09-23 (amended); revisited 2026-10-09 |
| D2 | Where unit handling lives | core parses `pint`; core takes plain numbers | Core takes a `PixelSize` in µm. `pint` stays in the plugin. Results don't change, because only pixel-size ratios are used, and Voxel Size is in working-grid pixels. | Accepted, 2026-09-23 |
| D3 | Error reporting | string sentinels; `RuntimeError`; own exception classes | `ClemregError(Exception)` subclasses, never `RuntimeError` (#25, #50, #53). On its own this doesn't fully fix #53: the adapters must also stop other `RuntimeError`s reaching generator workers. | Accepted, 2026-09-23 |
| D4 | Old import paths | break them; keep re-exports | Re-export with a `DeprecationWarning` until 1.0.0, or for at least six months after the split is released, whichever is later. Amended from "one release": copies of the published batch notebook won't upgrade in step. Needs the upstream owner's agreement before phase 2. | Accepted, 2026-09-23 (amended) |
| D5 | Where sample data lives | plugin; a napari-free `clemreg.data` | `clemreg.data`, returning arrays and a `PixelSize`; the plugin wraps them as layer tuples. Reversed from "plugin": the headless batch notebook uses the sample data, and `sample_data.py` has no napari imports anyway. The #42 metadata rework fits either way. | Accepted, 2026-09-23 (reversed) |
| D6 | Order of the split and MoBIE | MoBIE first (old plan); split first | Split first, then MoBIE directly in the core, released as `clemreg 0.2.0`. MoBIE can start once phase 1 merges; it doesn't wait for phases 2–4. | Accepted, 2026-09-23 |
| D7 | Keep the split widgets? | retire them in favour of Run Registration's step buttons; keep them | Out of scope for the split. They share the adapters either way (#35). | Open |
| D8 | Widget framework | keep magicgui; rewrite the widgets in raw Qt (as AIoD's plugin does) | Out of scope for the split. The adapters don't depend on either, so this can be decided later (see ROADMAP.md). | Open |
| D9 | Where point cloud sampling lives (#75) | (a) `clemreg.sampling` in the core, with the core's heavy dependencies (open3d, probreg) moved to an extra so the base install is light; (b) a small separate distribution the core depends on | Undecided. Either way, the samplers need only numpy, scipy and scikit-image, and the default edge method must keep the output baseline unchanged. | Open |
| D10 | Three-axis EM prediction (#48) | hide it unless `empanada-dl` is selected; pass it through to Segment-Flow; remove it | Removed: `segment_em` has no `three_axis` parameter, and neither widget shows the option. The end-to-end pipeline doesn't need it, and more powerful segmentation options are available in other plugins. `empanada_segmentation(axis_prediction=...)` stays, because the published batch notebook calls it. | Accepted, 2026-09-24 |
