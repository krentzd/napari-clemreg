#!/usr/bin/env python3
# coding: utf-8
import json
from pathlib import Path

import napari
import numpy as np
import pint
from magicgui import magic_factory, widgets
from napari.layers import Image, Shapes, Labels, Points
from napari.utils.notifications import show_error, show_info
from napari.qt.threading import GeneratorWorker
from ..clemreg.on_init_specs import specs
from ..clemreg._qt_layout import (
    extract_field_widget, group_into_collapsible, make_collapsible, refit_collapsible,
    wrap_in_scroll_area,
)

# The settings Save parameters writes and Parameters from JSON reads back.
# Pixel sizes, the ROI's z-range and the layers belong to the images.
SAVED_PARAMETERS = (
    'registration_algorithm', 'em_segmentation_backend',
    'log_sigma', 'log_threshold', 'filter_segmentation', 'filter_size_lower', 'filter_size_upper',
    'point_cloud_sampling_frequency', 'registration_voxel_size', 'point_cloud_sigma',
    'registration_max_iterations',
    'warping_interpolation_order', 'warping_approximate_grid', 'warping_sub_division_factor',
    'warping_output_resolution', 'visualise_intermediate_results', 'registration_direction',
)


def _save_parameters(path, settings):
    """Write the saved parameters among `settings` (a name -> value dict) to
    `path` as JSON, and say where. Shows an error and returns False if that
    isn't possible."""
    if path is None or Path(path).is_dir():
        show_error('Choose a file to save the parameters to (Path to save parameters).')
        return False
    parameters = {name: settings[name] for name in SAVED_PARAMETERS}
    path = Path(path).resolve()
    try:
        path.write_text(json.dumps(parameters, indent=4, ensure_ascii=False))
    except OSError as err:
        show_error(f"Couldn't save the parameters to {path}: {err}")
        return False
    show_info(f'Saved the parameters to {path}')
    return True


def _load_parameters(widget, path):
    """Fill the widget's fields from a file written by Save parameters.
    Settings the widget no longer has are ignored; if the file can't be
    used, nothing changes."""
    try:
        parameters = json.loads(Path(path).read_text())
        if not isinstance(parameters, dict):
            raise ValueError('expected a JSON object')
    except (OSError, ValueError) as err:
        show_error(f"Couldn't read {path} as a JSON parameter file: {err}")
        return
    previous = {name: getattr(widget, name).value for name in SAVED_PARAMETERS}
    for name in SAVED_PARAMETERS:
        if name not in parameters:
            continue
        try:
            getattr(widget, name).value = parameters[name]
        except (ValueError, TypeError) as err:
            for restored, value in previous.items():
                getattr(widget, restored).value = value
            show_error(f"Couldn't load the parameters in {path}: "
                       f"{name} = {parameters[name]!r} isn't valid ({err}).")
            return
from ..clemreg.exceptions import ClemregError


def _surface_runtime_errors(generator_function):
    """Re-raise any RuntimeError from a generator worker as a ClemregError.

    Register's segmentation and registration workers are generators, so
    thread_worker makes them superqt GeneratorWorkers. Those treat a
    RuntimeError as "the worker object was deleted" and swallow it,
    emitting neither `errored` nor `finished`, so Register silently hung
    (#53). RuntimeErrors can come from anywhere in the pipeline (numpy,
    open3d, probreg, torch), so convert them here. Apply it under
    @thread_worker.
    """
    import functools

    @functools.wraps(generator_function)
    def wrapper(*args, **kwargs):
        try:
            return (yield from generator_function(*args, **kwargs))
        except RuntimeError as exc:
            raise ClemregError(f'{type(exc).__name__}: {exc}') from exc

    return wrapper

# Use as worker.join workaround --> Launch registration thread_worker from here
class RegistrationThreadJoiner:
    def __init__(self, worker_function, init_kwargs, returned, yielded):
        self.moving_ready = False
        self.fixed_ready = False
        self.worker_function = worker_function
        self.init_kwargs = init_kwargs
        self.returned = returned
        self.yielded = yielded
        # Only ever populated by set_moving_kwargs/set_fixed_kwargs, which
        # are wired to each worker's `returned` signal -- i.e. only on
        # success. `finished` (see finished_fixed/finished_moving below)
        # fires whether the worker succeeded or errored, so None here is
        # the signal that a segmentation step failed rather than simply
        # "not started yet".
        self.moving_kwargs = None
        self.fixed_kwargs = None

    def set_moving_kwargs(self, kwargs):
        self.moving_kwargs = kwargs

    def set_fixed_kwargs(self, kwargs):
        self.fixed_kwargs = kwargs

    def finished_fixed(self):
        self.fixed_ready = True
        if self.moving_ready and self.fixed_ready:
            self.launch_worker()

    def finished_moving(self):
        self.moving_ready = True
        if self.moving_ready and self.fixed_ready:
            self.launch_worker()

    def launch_worker(self):
        # Confirmed bug (not new): `finished` fires on error too, so
        # without this check, a failed FM or EM segmentation left this
        # method reading self.moving_kwargs/self.fixed_kwargs that were
        # never set, crashing with an unrelated-looking AttributeError
        # inside a Qt slot -- which produced no further GUI activity at
        # all, looking exactly like the whole app had silently hung,
        # even though the real failure (and its traceback) had already
        # been reported separately via each worker's own `errored` signal.
        if self.moving_kwargs is None or self.fixed_kwargs is None:
            print('Registration not started: FM and/or EM segmentation did not complete successfully.')
            return

        print('Launching registration and warping...')

        worker = self.worker_function(**{**self.init_kwargs, **self.moving_kwargs,**self.fixed_kwargs})
        worker.returned.connect(self.returned)
        if isinstance(worker, GeneratorWorker):
            worker.yielded.connect(self.yielded)
        worker.start()

def on_init(widget):
    """ Initializes widget layout and updates widget layout according to user input.

    Parameters
    ----------
    widget : magicgui.widgets.Widget
        The parent widget of the plugin.
    """
    from ..clemreg.data_preprocessing import get_pixelsize

    json_settings = ['load_json_file']
    filter_segmentation_settings = ['filter_size_lower', 'filter_size_upper']
    save_json_settings = ['save_json_path']

    for x in json_settings + filter_segmentation_settings + save_json_settings + ['z_min', 'z_max']:
        setattr(getattr(widget, x), 'visible', False)

    # These used to be one flat list of ~20 widgets shown/hidden together
    # behind a single "Parameters custom" checkbox -- confirmed directly
    # that revealing them all in one burst was a real, several-second-plus
    # Qt style-polish/layout cost (worse than the isolated per-widget
    # cost, which profiles at under 10ms), and that the panel had no
    # bound on how tall it could grow as a result. Splitting them into
    # independent, always-present QCollapsible sections (collapsed by
    # default) means expanding one group only lays out that group's own
    # handful of widgets -- matching AIoD's own napari plugin, verified
    # against its real source, not guessed. The panel's overall height is
    # separately bounded by wrap_in_scroll_area() in
    # make_run_registration_widget() below.
    # Each section's own "Run this step" button (added below the section's
    # existing parameters) runs just that stage using whichever layers
    # currently sit in its inputs -- auto-set from a previous step, picked
    # manually, or pre-existing -- independent of the main "Register" call
    # button, which always runs the full pipeline from Moving_Image /
    # Fixed_Image onward. See on_init_specs.py's run_*_button entries and
    # the click handlers wired below.
    group_into_collapsible(widget, 'EM Segmentation Parameters',
                           ['em_segmentation_backend', 'run_em_segmentation_button'])
    group_into_collapsible(widget, 'LoG Segmentation Parameters',
                           ['log_sigma', 'log_threshold', 'filter_segmentation',
                            'filter_size_lower', 'filter_size_upper', 'run_fm_segmentation_button'])
    group_into_collapsible(widget, 'Point Cloud Sampling',
                           ['Moving_Segmentation', 'Fixed_Segmentation',
                            'point_cloud_sampling_frequency', 'point_cloud_sigma',
                            'run_point_cloud_sampling_button'])
    # The registration step's button also warps, so the warping settings
    # sit inside its section, above the button.
    warping = make_collapsible(widget, 'Image Warping',
                               ['warping_interpolation_order', 'warping_output_resolution',
                                'warping_approximate_grid', 'warping_sub_division_factor'])
    registration = group_into_collapsible(widget, 'Point Cloud Registration',
                                          ['Moving_Points', 'Fixed_Points',
                                           'registration_voxel_size', 'registration_max_iterations'])
    registration.addWidget(warping)
    registration.addWidget(extract_field_widget(widget, 'run_registration_and_warping_button'))

    # Cheap insurance for the same "doesn't shrink back down" behaviour
    # on the few small toggles left (1-2 widgets each) -- harmless even
    # now that the whole panel is scroll-bounded, and keeps the panel's
    # own content compact within the scroll area rather than leaving a
    # gap.
    def _shrink_to_fit():
        widget.native.adjustSize()

    def toggle_json_widget(load_json: bool):
        for x in json_settings:
            setattr(getattr(widget, x), 'visible', load_json)
        _shrink_to_fit()

    def toggle_filter_segmentation(filter_segmentation: bool):
        for x in filter_segmentation_settings:
            setattr(getattr(widget, x), 'visible', filter_segmentation)
        _shrink_to_fit()

    def toggle_save_json(save_json: bool):
        for x in save_json_settings:
            setattr(getattr(widget, x), 'visible', save_json)
        _shrink_to_fit()

    def show_bcpd_warping_settings(algorithm: str):
        # Only BCPD's non-rigid warp uses these; Rigid/Affine CPD warp with a matrix.
        for name in ('warping_approximate_grid', 'warping_sub_division_factor'):
            getattr(widget, name).visible = algorithm == 'BCPD'
        refit_collapsible(warping)
        _shrink_to_fit()

    def change_z_max(input_image: Image):
        if len(input_image.data.shape) == 3:
            widget.z_max.max = input_image.data.shape[0]
            widget.z_max.value = input_image.data.shape[0]
        elif len(input_image.data.shape) == 4:
            widget.z_max.max = input_image.data.shape[1]
            widget.z_max.value = input_image.data.shape[1]

    def change_z_min(z_max_val: int):
        widget.z_min.max = z_max_val

    def change_z_max_from_z_min(z_min_val: int):
        widget.z_max.min = z_min_val

    # TODO: z_min and z_max only shown and not hidden if no layer chosen
    def reveal_z_min_and_z_max():
        if len(widget.Mask_ROI.choices) > 0:
            for x in ['z_min', 'z_max']:
                setattr(getattr(widget, x), 'visible', True)
        else:
            for x in ['z_min', 'z_max']:
                setattr(getattr(widget, x), 'visible', False)

    def change_moving_pixelsize(input_image: Image):
        moving_xy_pixelsize, __, moving_z_pixelsize, unit = get_pixelsize(input_image.metadata)

        if unit in ['nanometer', 'nm', 'um', 'micron', 'micrometer']:
            if unit == 'um' or unit == 'micron':
                unit = 'micrometer'
            elif unit == 'nm':
                unit = 'nanometer'
        else:
            unit = 'nanometer'

        widget.moving_image_pixelsize_xy.value = str(moving_xy_pixelsize) + str(unit)
        widget.moving_image_pixelsize_z.value = str(moving_z_pixelsize) + str(unit)

    def change_fixed_pixelsize(input_image: Image):
        fixed_xy_pixelsize, __, fixed_z_pixelsize, unit = get_pixelsize(input_image.metadata)

        if unit in ['nanometer', 'nm', 'um', 'micron', 'micrometer']:
            if unit == 'um' or unit == 'micron':
                unit = 'micrometer'
            elif unit == 'nm':
                unit = 'nanometer'
        else:
            unit = 'nanometer'

        widget.fixed_image_pixelsize_xy.value = str(fixed_xy_pixelsize) + str(unit)
        widget.fixed_image_pixelsize_z.value = str(fixed_z_pixelsize) + str(unit)

    def keep_moving_and_fixed_apart():
        # Both dropdowns default to the first image layer, so with the
        # sample data both would be EM. Point Moving at another layer.
        fixed = widget.Fixed_Image.value
        if fixed is None or widget.Moving_Image.value is not fixed:
            return
        others = [layer for layer in widget.Moving_Image.choices if layer is not fixed]
        if others:
            widget.Moving_Image.value = others[0]

    def images_apart():
        moving, fixed = widget.Moving_Image.value, widget.Fixed_Image.value
        return moving is not None and fixed is not None and moving is not fixed

    def on_layer_inserted(event):
        # Usually the widget is docked before the data is opened. Stops once
        # the two differ, so later layers never override a user's choice.
        widget.Fixed_Image.reset_choices()
        widget.Moving_Image.reset_choices()
        keep_moving_and_fixed_apart()
        if images_apart():
            viewer.layers.events.inserted.disconnect(on_layer_inserted)

    widget.z_max.changed.connect(change_z_min)
    widget.Moving_Image.changed.connect(change_z_max)
    widget.Moving_Image.changed.connect(change_moving_pixelsize)
    widget.Fixed_Image.changed.connect(change_fixed_pixelsize)
    widget.z_min.changed.connect(change_z_max_from_z_min)
    widget.Mask_ROI.changed.connect(reveal_z_min_and_z_max)
    def load_parameters(*_):
        path = widget.load_json_file.value
        if widget.params_from_json.value and path is not None and Path(path).is_file():
            _load_parameters(widget, path)

    widget.params_from_json.changed.connect(toggle_json_widget)
    widget.params_from_json.changed.connect(load_parameters)
    widget.load_json_file.changed.connect(load_parameters)
    widget.filter_segmentation.changed.connect(toggle_filter_segmentation)
    widget.save_json.changed.connect(toggle_save_json)
    widget.registration_algorithm.changed.connect(show_bcpd_warping_settings)
    show_bcpd_warping_settings(widget.registration_algorithm.value)

    # A layer already selected at construction time (e.g. auto-selected
    # because it's the only choice, or the EM layer matching by name)
    # never fires its own `changed` signal -- nothing actually changed
    # from the widget's point of view -- so pixel sizes stayed at their
    # "0 nanometer" placeholder until the user manually reselected a
    # layer. Confirmed directly: this is exactly what was happening with
    # the EM layer, which starts pre-selected as the only available
    # choice. Seed both fields once here to match what's already shown.
    keep_moving_and_fixed_apart()
    if widget.Moving_Image.value is not None:
        change_moving_pixelsize(widget.Moving_Image.value)
    if widget.Fixed_Image.value is not None:
        change_fixed_pixelsize(widget.Fixed_Image.value)

    from napari.viewer import current_viewer
    viewer = current_viewer()
    if viewer is not None and not images_apart():
        viewer.layers.events.inserted.connect(on_layer_inserted)

    _wire_step_buttons(widget)

def _wire_step_buttons(widget):
    """Wire each collapsible section's "Run this step" button to run just
    that stage, using whichever layers currently sit in its own inputs.

    Each handler reads current values directly off `widget.<field>.value`
    rather than through make_run_registration()'s own parameters, since
    on_init() (where these are wired) runs once at widget construction --
    before the decorated function has ever been called with any
    arguments -- so there's no shared closure to reuse; the actual
    pipeline logic still lives only in clemreg/widget_components.py,
    called identically from here and from make_run_registration()'s body.
    """
    from napari.qt.threading import thread_worker
    from napari.viewer import current_viewer
    from ..clemreg._qt_layout import mark_auto_set, clear_highlight_on_user_select

    def _with_button_reset(button, fn):
        """Wrap a worker's returned/errored callback to re-enable `button`
        first -- matches AIoD's own aiod_napari (nxf.py's nxf_run_btn)
        pattern of disabling a run button for the duration of its own
        run, rather than leaving every section runnable mid-pipeline.
        """
        def wrapped(*args):
            button.native.setEnabled(True)
            fn(*args)
        return wrapped

    # Applies regardless of whether a downstream field's value came from
    # a single section's own "Run this step" button or from the full
    # "Register" run -- the auto-set highlight means the same thing
    # either way.
    for field_name in ('Fixed_Segmentation', 'Moving_Segmentation', 'Moving_Points', 'Fixed_Points'):
        clear_highlight_on_user_select(getattr(widget, field_name))

    # make_run_registration()'s own body has no reference back to this
    # Container (magicgui calls the decorated function as a plain
    # function, with no self-injection -- confirmed directly), so its
    # internal thread-worker callbacks can't call mark_auto_set()
    # themselves. Watching the viewer's own layer-inserted event instead
    # means the "Register" flow needs no changes at all to get the same
    # highlighting: match by the fixed layer names widget_components.py
    # already uses.
    _PIPELINE_OUTPUT_FIELDS = {
        'FM_segmentation': 'Moving_Segmentation',
        'EM_segmentation': 'Fixed_Segmentation',
        'Moving_point_cloud': 'Moving_Points',
        'Fixed_point_cloud': 'Fixed_Points',
    }

    def _highlight_if_pipeline_output(event):
        layer = event.value
        field_name = _PIPELINE_OUTPUT_FIELDS.get(layer.name)
        if field_name is not None:
            mark_auto_set(getattr(widget, field_name), layer)

    viewer = current_viewer()
    if viewer is not None:
        viewer.layers.events.inserted.connect(_highlight_if_pipeline_output)

    def _run_em_segmentation_step():
        from ..clemreg.widget_components import em_backend_ready, run_fixed_segmentation

        fixed_image = widget.Fixed_Image.value
        backend = widget.em_segmentation_backend.value
        if fixed_image is None:
            show_error("WARNING: You have not inputted a Fixed Image")
            return
        if len(fixed_image.data.shape) != 3:
            show_error("WARNING: Your Fixed_Image must be 3D, your current input has a shape of {}".format(
                fixed_image.data.shape))
            return
        if not em_backend_ready(backend):
            return

        def _done(layer):
            viewer = current_viewer()
            if viewer is not None:
                viewer.add_layer(layer)
            mark_auto_set(widget.Fixed_Segmentation, layer)

        def _errored(exc):
            show_error(f'EM segmentation failed: {exc}')

        button = widget.run_em_segmentation_button
        button.native.setEnabled(False)

        @thread_worker(connect={
            'returned': _with_button_reset(button, _done),
            'errored': _with_button_reset(button, _errored),
        })
        def _thread():
            seg_volume = run_fixed_segmentation(Fixed_Image=fixed_image, em_segmentation_backend=backend)
            return Labels(seg_volume.astype(np.int64), name='EM_segmentation', metadata=fixed_image.metadata)

        _thread()

    def _run_fm_segmentation_step():
        from ..clemreg.widget_components import run_moving_segmentation
        from ..clemreg.mask_roi import mask_area

        moving_image = widget.Moving_Image.value
        mask_roi = widget.Mask_ROI.value
        if moving_image is None:
            show_error("WARNING: You have not inputted a Moving Image")
            return
        if len(moving_image.data.shape) != 3:
            show_error("WARNING: Your Moving_Image must be 3D, your current input has a shape of {}".format(
                moving_image.data.shape))
            return
        if mask_roi is not None:
            if len(mask_roi.data) != 1:
                show_error("WARNING: You must input only 1 Mask ROI, you have inputted {}.".format(
                    len(mask_roi.data)))
                return
            if mask_area(mask_roi.data[0][:, 1], mask_roi.data[0][:, 2]) > \
                    moving_image.data.shape[1] * moving_image.data.shape[2]:
                show_error("WARNING: Your mask size exceeds the size of the image.")
                return

        def _done(layer):
            viewer = current_viewer()
            if viewer is not None:
                viewer.add_layer(layer)
            mark_auto_set(widget.Moving_Segmentation, layer)

        def _errored(exc):
            show_error(f'FM segmentation failed: {exc}')

        button = widget.run_fm_segmentation_button
        button.native.setEnabled(False)

        @thread_worker(connect={
            'returned': _with_button_reset(button, _done),
            'errored': _with_button_reset(button, _errored),
        })
        def _thread():
            seg_volume_mask = run_moving_segmentation(Moving_Image=moving_image,
                                                       Mask_ROI=mask_roi,
                                                       z_min=widget.z_min.value,
                                                       z_max=widget.z_max.value,
                                                       log_sigma=widget.log_sigma.value,
                                                       log_threshold=widget.log_threshold.value,
                                                       filter_segmentation=widget.filter_segmentation.value,
                                                       filter_size_lower=widget.filter_size_lower.value,
                                                       filter_size_upper=widget.filter_size_upper.value)
            return Labels(seg_volume_mask.astype(np.uint32), name='FM_segmentation', metadata=moving_image.metadata)

        _thread()

    def _run_point_cloud_sampling_step():
        from ..clemreg.widget_components import run_point_cloud_sampling

        moving_segmentation = widget.Moving_Segmentation.value
        fixed_segmentation = widget.Fixed_Segmentation.value
        if moving_segmentation is None or fixed_segmentation is None:
            show_error("WARNING: You have not inputted both a Moving_Segmentation and Fixed_Segmentation")
            return

        def _done(result):
            moving_points, fixed_points = result
            viewer = current_viewer()
            if viewer is not None:
                viewer.add_layer(moving_points)
                viewer.add_layer(fixed_points)
            mark_auto_set(widget.Moving_Points, moving_points)
            mark_auto_set(widget.Fixed_Points, fixed_points)

        def _errored(exc):
            show_error(f'Point cloud sampling failed: {exc}')

        button = widget.run_point_cloud_sampling_button
        button.native.setEnabled(False)

        @thread_worker(connect={
            'returned': _with_button_reset(button, _done),
            'errored': _with_button_reset(button, _errored),
        })
        def _thread():
            return run_point_cloud_sampling(
                Moving_Segmentation=moving_segmentation,
                Fixed_Segmentation=fixed_segmentation,
                moving_image_pixelsize_xy=widget.moving_image_pixelsize_xy.value,
                moving_image_pixelsize_z=widget.moving_image_pixelsize_z.value,
                fixed_image_pixelsize_xy=widget.fixed_image_pixelsize_xy.value,
                fixed_image_pixelsize_z=widget.fixed_image_pixelsize_z.value,
                point_cloud_sampling_frequency=widget.point_cloud_sampling_frequency.value,
                voxel_size=widget.registration_voxel_size.value,
                point_cloud_sigma=widget.point_cloud_sigma.value)

        _thread()

    def _run_registration_and_warping_step():
        from ..clemreg.widget_components import run_point_cloud_registration_and_warping
        from ..clemreg._napari_compat import link_layers

        moving_points = widget.Moving_Points.value
        fixed_points = widget.Fixed_Points.value
        moving_image = widget.Moving_Image.value
        fixed_image = widget.Fixed_Image.value
        if moving_points is None or fixed_points is None:
            show_error("WARNING: You have not inputted both a Moving_Points and Fixed_Points")
            return
        if moving_image is None or fixed_image is None:
            show_error("WARNING: You have not inputted both a Moving_Image and Fixed_Image")
            return
        # This step completes a registration, so it saves like Register does.
        if widget.save_json.value:
            settings = {name: getattr(widget, name).value for name in SAVED_PARAMETERS}
            if not _save_parameters(widget.save_json_path.value, settings):
                return

        def _done(result):
            warp_outputs, transformed = result
            viewer = current_viewer()
            if viewer is None:
                return
            layers = []
            for image_layer in warp_outputs:
                viewer.add_layer(image_layer)
                layers.append(viewer.layers[image_layer.name])
            link_layers(layers)

        def _errored(exc):
            show_error(f'Registration/warping failed: {exc}')

        button = widget.run_registration_and_warping_button
        button.native.setEnabled(False)

        @thread_worker(connect={
            'returned': _with_button_reset(button, _done),
            'errored': _with_button_reset(button, _errored),
        })
        def _thread():
            return run_point_cloud_registration_and_warping(
                Moving_Points=moving_points,
                Fixed_Points=fixed_points,
                Moving_Image=moving_image,
                Fixed_Image=fixed_image,
                registration_algorithm=widget.registration_algorithm.value,
                registration_max_iterations=widget.registration_max_iterations.value,
                warping_interpolation_order=widget.warping_interpolation_order.value,
                warping_approximate_grid=widget.warping_approximate_grid.value,
                warping_sub_division_factor=widget.warping_sub_division_factor.value,
                warping_output_resolution=widget.warping_output_resolution.value,
                registration_direction=widget.registration_direction.value)

        _thread()

    widget.run_em_segmentation_button.clicked.connect(_run_em_segmentation_step)
    widget.run_fm_segmentation_button.clicked.connect(_run_fm_segmentation_step)
    widget.run_point_cloud_sampling_button.clicked.connect(_run_point_cloud_sampling_step)
    widget.run_registration_and_warping_button.clicked.connect(_run_registration_and_warping_step)

@magic_factory(widget_init=on_init, layout='vertical', call_button='Register',
               widget_header={'widget_type': 'Label',
                              'label': f'<h1 text-align="left">CLEM-Reg</h1>'},

               z_min=specs['z_min'],
               z_max=specs['z_max'],
               registration_algorithm=specs['registration_algorithm'],
               params_from_json=specs['params_from_json'],
               load_json_file=specs['load_json_file'],

               em_segmentation_backend=specs['em_segmentation_backend'],

               log_sigma=specs['log_sigma'],
               log_threshold=specs['log_threshold'],
               filter_segmentation=specs['filter_segmentation'],
               filter_size_lower=specs['filter_size_lower'],
               filter_size_upper=specs['filter_size_upper'],

               point_cloud_sampling_frequency=specs['point_cloud_sampling_frequency'],
               point_cloud_sigma=specs['point_cloud_sigma'],

               registration_voxel_size=specs['registration_voxel_size'],
               registration_max_iterations=specs['registration_max_iterations'],

               warping_interpolation_order=specs['warping_interpolation_order'],
               warping_approximate_grid=specs['warping_approximate_grid'],
               warping_sub_division_factor=specs['warping_sub_division_factor'],
               warping_output_resolution=specs['warping_output_resolution'],
               save_json=specs['save_json'],
               save_json_path=specs['save_json_path'],
               visualise_intermediate_results=specs['visualise_intermediate_results'],
               moving_image_pixelsize_xy=specs['moving_image_pixelsize_xy'],
               moving_image_pixelsize_z=specs['moving_image_pixelsize_z'],
               fixed_image_pixelsize_xy=specs['fixed_image_pixelsize_xy'],
               fixed_image_pixelsize_z=specs['fixed_image_pixelsize_z'],
               registration_direction=specs['registration_direction'],
               Moving_Image=specs['Moving_Image'],
               Fixed_Image=specs['Fixed_Image'],

               Moving_Segmentation=specs['Moving_Segmentation'],
               Fixed_Segmentation=specs['Fixed_Segmentation'],
               Moving_Points=specs['Moving_Points'],
               Fixed_Points=specs['Fixed_Points'],

               run_em_segmentation_button=specs['run_em_segmentation_button'],
               run_fm_segmentation_button=specs['run_fm_segmentation_button'],
               run_point_cloud_sampling_button=specs['run_point_cloud_sampling_button'],
               run_registration_and_warping_button=specs['run_registration_and_warping_button'],
               )
def make_run_registration(
        viewer: 'napari.viewer.Viewer',
        widget_header,
        Moving_Image: Image,

        moving_image_pixelsize_xy,
        moving_image_pixelsize_z,
        Mask_ROI: Shapes,
        z_min,
        z_max,

        Fixed_Image: Image,
        fixed_image_pixelsize_xy,
        fixed_image_pixelsize_z,

        registration_algorithm,
        params_from_json,
        load_json_file,

        em_segmentation_backend,
        run_em_segmentation_button,

        log_sigma,
        log_threshold,
        filter_segmentation,
        filter_size_lower,
        filter_size_upper,
        run_fm_segmentation_button,

        Moving_Segmentation: Labels,
        Fixed_Segmentation: Labels,
        point_cloud_sampling_frequency,
        registration_voxel_size,
        point_cloud_sigma,
        run_point_cloud_sampling_button,

        Moving_Points: Points,
        Fixed_Points: Points,
        registration_max_iterations,
        run_registration_and_warping_button,

        warping_interpolation_order,
        warping_approximate_grid,
        warping_sub_division_factor,
        warping_output_resolution,

        save_json,
        save_json_path,
        visualise_intermediate_results,

        registration_direction
        ) -> Image:
    """Run CLEM-Reg end-to-end

    Parameters
    ----------
    save_json
    viewer
    widget_header
    Moving_Image
    Fixed_Image
    Mask_ROI
    z_min
    z_max
    registration_algorithm
    em_segmentation_backend
    log_sigma
    log_threshold
    zoom_value
    filter_size
    point_cloud_sampling_frequency
    point_cloud_sigma
    registration_voxel_size
    registration_max_iterations
    warping_interpolation_order
    warping_approximate_grid
    warping_sub_division_factor
    warping_output_resolution
    Moving_Segmentation
    Fixed_Segmentation
    Moving_Points
    Fixed_Points

    Returns
    -------

    """
    # First, before any other local name exists: Save parameters writes these.
    arguments = dict(locals())
    import time
    from pathlib import Path
    from ..clemreg.log_segmentation import log_segmentation, filter_binary_segmentation
    from ..clemreg.mask_roi import mask_roi, mask_area
    from ..clemreg.point_cloud_registration import point_cloud_registration
    from ..clemreg.point_cloud_sampling import point_cloud_sampling
    from ..clemreg.warp_image_volume import warp_image_volume
    from ..clemreg.data_preprocessing import make_isotropic, _make_isotropic
    from napari.qt.threading import thread_worker
    from ..clemreg._napari_compat import link_layers

    def _add_data(return_value):
        if isinstance(return_value, list):
            layers = []
            for image_layer in return_value:
                print(f'Adding {image_layer.name} to viewer...')
                viewer.add_layer(image_layer)
                layers.append(viewer.layers[image_layer.name])
            link_layers(layers)
            # This return_value (a list of warped Image layers) is only
            # ever produced once, by the final stage of the pipeline --
            # so its arrival marks the whole "Register" run as complete,
            # timed from just before the moving/fixed segmentation
            # workers below were started. mm:ss, not raw seconds -- a
            # full run is realistically minutes, not sub-minute.
            elapsed = int(time.time() - start_time)
            print(f'Run Registration finished in {elapsed // 60:02d}:{elapsed % 60:02d}')
        else:
            print(f'Adding {return_value.name} to viewer...')
            viewer.add_layer(return_value)

    def _yield_segmentation(yield_value):
        viewer.add_layer(yield_value)

    def _yield_point_clouds(yield_value):
        points, kwargs = yield_value
        # Keep the affine: Rigid and Affine CPD's transformed points are the
        # moving points, with the registration as the layer's affine.
        viewer.add_points(points.data, affine=points.affine, **kwargs)

    @thread_worker
    @_surface_runtime_errors
    def _run_moving_thread(**kwargs):
        from ..clemreg.widget_components import run_moving_segmentation

        # Raises NoSegmentationError if nothing is found (#25); the
        # errored handler below reports it.
        seg_volume_mask = run_moving_segmentation(**kwargs)
        seg_volume_mask = Labels(seg_volume_mask.astype(np.uint32),
                                 name='FM_segmentation',
                                 metadata=Moving_Image.metadata)

        if visualise_intermediate_results:
            yield seg_volume_mask

        return {'Moving_Segmentation': seg_volume_mask}

    @thread_worker
    @_surface_runtime_errors
    def _run_fixed_thread(**kwargs):
        from ..clemreg.widget_components import run_fixed_segmentation
        #Increasing levels of CLAHE

        # Raises NoSegmentationError if nothing is found (#25).
        seg_volume = run_fixed_segmentation(**kwargs)
        seg_volume = Labels(seg_volume.astype(np.int64),
                            name='EM_segmentation',
                            metadata=Fixed_Image.metadata)

        if visualise_intermediate_results:
            yield seg_volume

        return {'Fixed_Segmentation': seg_volume}

    @thread_worker
    @_surface_runtime_errors
    def _run_registration_thread(**kwargs):
        from ..clemreg.widget_components import run_point_cloud_sampling
        from ..clemreg.widget_components import run_point_cloud_registration_and_warping

        point_cloud_keys = ['Moving_Segmentation',
                            'Fixed_Segmentation',
                            'moving_image_pixelsize_xy',
                            'moving_image_pixelsize_z',
                            'fixed_image_pixelsize_xy',
                            'fixed_image_pixelsize_z',
                            'point_cloud_sampling_frequency',
                            'voxel_size',
                            'point_cloud_sigma']

        point_cloud_kwargs = dict((k, kwargs[k]) for k in point_cloud_keys if k in kwargs)
        moving_points, fixed_points = run_point_cloud_sampling(**point_cloud_kwargs)

        if visualise_intermediate_results:
            yield (moving_points, {'name': 'moving_points', 'face_color': 'red'})
            yield (fixed_points, {'name': 'fixed_points', 'face_color': 'blue'})

        reg_and_warping_keys = ['Moving_Image',
                                'Fixed_Image',
                                'registration_algorithm',
                                'registration_max_iterations',
                                'warping_interpolation_order',
                                'warping_approximate_grid',
                                'warping_sub_division_factor',
                                'warping_output_resolution',
                                'registration_direction']

        reg_and_warping_kwargs = dict((k, kwargs[k]) for k in reg_and_warping_keys if k in kwargs)
        point_cloud_return_kwargs = dict(Moving_Points=moving_points, Fixed_Points=fixed_points)
        point_cloud_reg_and_warping_kwargs = {**point_cloud_return_kwargs, **reg_and_warping_kwargs}
        warp_outputs, transformed = run_point_cloud_registration_and_warping(**point_cloud_reg_and_warping_kwargs)

        if visualise_intermediate_results:
            yield (transformed, {'name': 'transformed_points', 'face_color': 'yellow'})

        return warp_outputs

    if Moving_Image is None or Fixed_Image is None:
        show_error("WARNING: You have not inputted both a fixed and moving image")
        return

    if len(Moving_Image.data.shape) != 3:
        show_error("WARNING: Your moving_image must be 3D, you're current input has a shape of {}".format(
            Moving_Image.data.shape))
        return
    elif len(Moving_Image.data.shape) == 3 and (Moving_Image.data.shape[2] == 3 or Fixed_Image.data.shape[2] == 4):
        show_error("WARNING: YOUR moving_image is RGB, your input must be grayscale and 3D")
        return

    if len(Fixed_Image.data.shape) != 3:
        show_error("WARNING: Your Fixed_Image must be 3D, you're current input has a shape of {}".format(
            Moving_Image.data.shape))
        return
    elif len(Fixed_Image.data.shape) == 3 and (Fixed_Image.data.shape[2] == 3 or Fixed_Image.data.shape[2] == 4):
        show_error("WARNING: YOUR fixed_image is RGB, your input must be grayscale and 3D")
        return

    if Mask_ROI is not None:
        if len(Mask_ROI.data) != 1:
            show_error("WARNING: You must input only 1 Mask ROI, you have inputted {}.".format(len(Mask_ROI.data)))
            return
        if mask_area(Mask_ROI.data[0][:, 1], Mask_ROI.data[0][:, 2]) > Moving_Image.data.shape[1] * \
                Moving_Image.data.shape[2]:
            show_error("WARNING: Your mask size exceeds the size of the image.")
            return

    from ..clemreg.widget_components import em_backend_ready

    if not em_backend_ready(em_segmentation_backend):
        return

    if save_json and not _save_parameters(save_json_path, arguments):
        return

    start_time = time.time()

    registration_thread_kwargs = dict(
        moving_image_pixelsize_xy=moving_image_pixelsize_xy,
        moving_image_pixelsize_z=moving_image_pixelsize_z,
        fixed_image_pixelsize_xy=fixed_image_pixelsize_xy,
        fixed_image_pixelsize_z=fixed_image_pixelsize_z,
        point_cloud_sampling_frequency=point_cloud_sampling_frequency,
        voxel_size=registration_voxel_size,
        point_cloud_sigma=point_cloud_sigma,
        Moving_Image=Moving_Image,
        Fixed_Image=Fixed_Image,
        registration_algorithm=registration_algorithm,
        registration_max_iterations=registration_max_iterations,
        warping_interpolation_order=warping_interpolation_order,
        warping_approximate_grid=warping_approximate_grid,
        warping_sub_division_factor=warping_sub_division_factor,
        warping_output_resolution=warping_output_resolution,
        registration_direction=registration_direction
    )
    joiner = RegistrationThreadJoiner(worker_function=_run_registration_thread,
                                      init_kwargs=registration_thread_kwargs,
                                      returned=_add_data,
                                      yielded=_yield_point_clouds)

    def _class_setter_moving(x):
        joiner.set_moving_kwargs(x)

    def _class_setter_fixed(x):
        joiner.set_fixed_kwargs(x)

    def _finished_moving_emitter():
        joiner.finished_moving()

    def _finished_fixed_emitter():
        joiner.finished_fixed()

    def _moving_segmentation_errored(exc):
        show_error(f'FM segmentation failed, registration will not run: {exc}')

    def _fixed_segmentation_errored(exc):
        show_error(f'EM segmentation failed, registration will not run: {exc}')

    worker_moving = _run_moving_thread(Moving_Image=Moving_Image,
                                       Mask_ROI=Mask_ROI,
                                       z_min=z_min,
                                       z_max=z_max,
                                       log_sigma=log_sigma,
                                       log_threshold=log_threshold,
                                       filter_segmentation=filter_segmentation,
                                       filter_size_lower=filter_size_lower,
                                       filter_size_upper=filter_size_upper)
    worker_moving.returned.connect(_class_setter_moving)
    worker_moving.finished.connect(_finished_moving_emitter)
    worker_moving.yielded.connect(_yield_segmentation)
    worker_moving.errored.connect(_moving_segmentation_errored)
    worker_moving.start()

    worker_fixed = _run_fixed_thread(Fixed_Image=Fixed_Image,
                                     em_segmentation_backend=em_segmentation_backend)
    worker_fixed.returned.connect(_class_setter_fixed)
    worker_fixed.finished.connect(_finished_fixed_emitter)
    worker_fixed.errored.connect(_fixed_segmentation_errored)
    worker_fixed.yielded.connect(_yield_segmentation)
    worker_fixed.start()


def make_run_registration_widget(napari_viewer: 'napari.viewer.Viewer' = None):
    """The actual napari-docked widget. Builds the same FunctionGui as
    ``make_run_registration()`` (kept as its own importable factory --
    unchanged for anything, such as the tests, that wants the raw
    magicgui object and its fields directly), then wraps its native
    widget in a QScrollArea so the dock panel has a bounded height no
    matter how many of the collapsible sections above are expanded at
    once -- matching AIoD's own napari plugin's approach (verified
    against its real source), rather than the ad hoc sizeHint/adjustSize
    patching this used before.
    """
    # napari_viewer defaults to None and is otherwise unused -- confirmed
    # directly in napari's own source (_qnpe2._get_widget_viewer_param)
    # that its viewer-auto-injection-by-parameter-name only applies to
    # class-based (QWidget/magicgui.widgets.Widget subclass) widgets; for
    # a plain function like this one it's unconditionally skipped ("For
    # magicgui type widget contributions, Viewer injection is done by
    # magicgui.register_type instead" -- that code's own comment), so
    # napari calls this with zero arguments. A required positional
    # parameter here crashed with exactly that TypeError.
    #
    # magic_factory's own __call__ also treats kwargs as widget-option
    # overrides, not runtime values -- confirmed directly
    # (`viewer=napari_viewer` raised "must be a dict"). So: call with no
    # args and let its own Viewer-typed-parameter auto-injection (via
    # magicgui's registered napari.viewer.Viewer type provider) resolve
    # it, exactly as it already did before this widget was wrapped.
    gui = make_run_registration()
    return wrap_in_scroll_area(gui)
