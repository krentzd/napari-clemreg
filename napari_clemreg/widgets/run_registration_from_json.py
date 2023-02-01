#!/usr/bin/env python3
# coding: utf-8
import json
import warnings
import napari
from magicgui import magic_factory
from napari.layers import Image, Shapes, Labels, Points
from pathlib import Path

# Use as worker.join workaround --> Launch registration thread_worker from here
class RegistrationThreadJoiner:
    def __init__(self, worker_function):
        self.moving_ready = False
        self.fixed_ready = False
        self.fixed_points = None
        self.moving_points = None
        self.worker_function = worker_function

    def set_fixed_points(self, points):
        self.fixed_points = points

    def set_moving_points(self, points):
        self.moving_points = points

    def finished_fixed(self):
        self.fixed_ready = True
        print('Fixed points set', self.fixed_ready, self.moving_ready)
        print(self.fixed_points, self.moving_points)
        if self.moving_ready and self.fixed_ready:
            self.launch_worker()

    def finished_moving(self):
        self.moving_ready = True
        print('Moving points set', self.fixed_ready, self.moving_ready)
        print(self.fixed_points, self.moving_points)
        if self.moving_ready and self.fixed_ready:
            self.launch_worker()

    def launch_worker(self):
        self.worker_function(self.moving_points, self.fixed_points)


def on_init(widget):
    """ Initializes widget layout and updates widget layout according to user input.

    Parameters
    ----------
    widget : magicgui.widgets.Widget
        The parent widget of the plugin.
    """
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

    widget.z_max.changed.connect(change_z_min)
    widget.Moving_Image.changed.connect(change_z_max)
    widget.z_min.changed.connect(change_z_max_from_z_min)
    widget.Mask_ROI.changed.connect(reveal_z_min_and_z_max)

    # widget.advanced.changed.connect(toggle_transform_widget)


@magic_factory(widget_init=on_init, layout='vertical', call_button='Register',
               widget_header={'widget_type': 'Label',
                              'label': f'<h1 text-align="left">CLEM-Reg</h1>'},

               z_min={'widget_type': 'SpinBox',
                      'label': 'Minimum z value for masking',
                      "min": 0, "max": 10, "step": 1,
                      'value': 0},
               z_max={'widget_type': 'SpinBox',
                      'label': 'Maximum z value for masking',
                      "min": 0, "max": 10, "step": 1,
                      'value': 0},
               )
def make_run_registration(
        viewer: 'napari.viewer.Viewer',
        widget_header,
        Moving_Image: Image,
        Fixed_Image: Image,
        Mask_ROI: Shapes,
        z_min,
        z_max,
        json_file: Path
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
    advanced
    white_space_0
    em_seg_header
    em_seg_axis
    white_space_1
    log_header
    log_sigma
    log_threshold
    white_space_2
    point_cloud_header
    point_cloud_sampling_frequency
    point_cloud_sigma
    white_space_3
    registration_header
    registration_voxel_size
    registration_every_k_points
    registration_max_iterations
    white_space_4
    warping_header
    warping_interpolation_order
    warping_approximate_grid
    warping_sub_division_factor

    Returns
    -------

    """
    from ..clemreg.empanada_segmentation import empanada_segmentation
    from ..clemreg.log_segmentation import log_segmentation
    from ..clemreg.mask_roi import mask_roi, mask_area
    from ..clemreg.point_cloud_registration import point_cloud_registration
    from ..clemreg.point_cloud_sampling import point_cloud_sampling
    from ..clemreg.warp_image_volume import warp_image_volume
    from napari.qt.threading import thread_worker

    if json_file.is_file() and '.json' in str(json_file):
        f = open(str(json_file))

        data = json.load(f)
        try:
            registration_algorithm = data["registration_algorithm"]
            em_seg_axis = data["em_seg_axis"]
            log_sigma = data["log_sigma"]
            log_threshold = data["log_threshold"]
            point_cloud_sampling_frequency = data["point_cloud_sampling_frequency"]
            point_cloud_sigma = data["point_cloud_sigma"]
            registration_voxel_size = data ["registration_voxel_size"]
            registration_every_k_points = data["registration_every_k_points"]
            registration_max_iterations = data["registration_max_iterations"]
            warping_interpolation_order = data["warping_interpolation_order"]
            warping_approximate_grid = data["warping_approximate_grid"]
            warping_sub_division_factor = data["warping_sub_division_factor"]
        except KeyError:
            warnings.warn("JSON file missing required param")
            return
    else:
        warnings.warn("Path to JSON file doesn't exist or file is not a JSON")
        return

    @thread_worker
    def _run_moving_thread():
        seg_volume = log_segmentation(input=Moving_Image,
                                      sigma=log_sigma,
                                      threshold=log_threshold)
        print('Mask_ROI:', Mask_ROI)
        if Mask_ROI is not None:
            seg_volume_mask = mask_roi(input=seg_volume,
                                       crop_mask=Mask_ROI, z_min=z_min, z_max=z_max)
        else:
            seg_volume_mask = seg_volume

        point_cloud = point_cloud_sampling(input=seg_volume_mask,
                                           sampling_frequency=point_cloud_sampling_frequency / 100,
                                           sigma=point_cloud_sigma)
        return point_cloud

    @thread_worker
    def _run_fixed_thread():
        seg_volume = empanada_segmentation(input=Fixed_Image.data,
                                           axis_prediction=em_seg_axis)
        print(seg_volume)
        point_cloud = point_cloud_sampling(input=Labels(seg_volume),
                                           sampling_frequency=point_cloud_sampling_frequency / 100,
                                           sigma=point_cloud_sigma)
        return point_cloud

    def _add_data(return_value):
        data, kwargs = return_value
        viewer.add_image(data, **kwargs)

    def _create_json_file():
        dictionary = {
            "registration_algorithm": registration_algorithm,
            "em_seg_axis": em_seg_axis,
            "log_sigma": log_sigma,
            "log_threshold": log_threshold,
            "point_cloud_sampling_frequency": point_cloud_sampling_frequency,
            "point_cloud_sigma": point_cloud_sigma,
            "registration_voxel_size": registration_voxel_size,
            "registration_every_k_points": registration_every_k_points,
            "registration_max_iterations": registration_max_iterations,
            "warping_interpolation_order": warping_interpolation_order,
            "warping_approximate_grid": warping_approximate_grid,
            "warping_sub_division_factor": warping_sub_division_factor
        }

        json_object = json.dumps(dictionary, indent=4)

        with open("sample.json", "w") as outfile:
            outfile.write(json_object)

    @thread_worker(connect={"returned": _add_data})
    def _run_registration_thread(moving_points, fixed_points):
        print('Entered thread!')

        moving, fixed, transformed, kwargs = point_cloud_registration(moving_points.data, fixed_points.data,
                                                                      algorithm=registration_algorithm,
                                                                      voxel_size=registration_voxel_size,
                                                                      every_k_points=registration_every_k_points,
                                                                      max_iterations=registration_max_iterations)

        if registration_algorithm == 'Affine CPD' or registration_algorithm == 'Rigid CPD':
            transformed = Points(moving, **kwargs)
            print(transformed)

        return warp_image_volume(moving_image=Moving_Image,
                                 fixed_image=Fixed_Image.data,
                                 transform_type=registration_algorithm,
                                 moving_points=Points(moving),
                                 transformed_points=transformed,
                                 interpolation_order=warping_interpolation_order,
                                 approximate_grid=warping_approximate_grid,
                                 sub_division_factor=warping_sub_division_factor)

    if Moving_Image is None or Fixed_Image is None:
        warnings.warn("WARNING: You have not inputted both a fixed and moving image")
        return

    if len(Moving_Image.data.shape) == 2 or len(Fixed_Image.data.shape) == 2:
        warnings.warn(
            "WARNING: Your input must be 3D, you're current input has a shape of {}".format(Moving_Image.data.shape))
        return

    if Moving_Image.data.shape != Fixed_Image.data.shape:
        warnings.warn(
            "WARNING: Your fixed and moving images must have the same shape. Fixed shape: {fixed_shape} != Moving "
            "shape:{moving_shape}".format(
                fixed_shape=Fixed_Image.data.shape,
                moving_shape=Moving_Image.data.shape))
        return

    if Mask_ROI is not None:
        if len(Mask_ROI.data) != 1:
            warnings.warn("WARNING: You must input only 1 Mask ROI, you have inputted {}.".format(len(Mask_ROI.data)))
            return
        if mask_area(Mask_ROI.data[0][:, 1], Mask_ROI.data[0][:, 2]) > Moving_Image.data.shape[1] * \
                Moving_Image.data.shape[2]:
            warnings.warn("WARNING: Your mask size exceeds the size of the image.")
            return

    joiner = RegistrationThreadJoiner(worker_function=_run_registration_thread)

    def _class_setter_moving(x):
        joiner.set_moving_points(x)

    def _class_setter_fixed(x):
        joiner.set_fixed_points(x)

    def _finished_moving_emitter():
        joiner.finished_moving()

    def _finished_fixed_emitter():
        joiner.finished_fixed()

    worker_moving = _run_moving_thread()
    worker_moving.returned.connect(_class_setter_moving)
    worker_moving.finished.connect(_finished_moving_emitter)
    worker_moving.start()

    worker_fixed = _run_fixed_thread()
    worker_fixed.returned.connect(_class_setter_fixed)
    worker_fixed.finished.connect(_finished_fixed_emitter)
    worker_fixed.start()
