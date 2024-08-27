import os
import re
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    Label,
    LineEdit,
    ProgressBar,
    PushButton,
    RadioButtons,
    Select,
    SpinBox,
    Table,
    TupleEdit,
)
from napari import layers
from pyclesperanto_prototype import set_wait_for_kernel_finish
from qtpy.QtWidgets import QTabWidget

from napari_ndev import helpers

if TYPE_CHECKING:
    import napari

class MeasureContainer(Container):
    """ Container class for measuring labels from folders. Can compare against
    intensity images, which can be microscopy data or other labels.
    """
    def __init__(
        self,
        viewer: "napari.viewer.Viewer" = None,
    ):
        super().__init__()

        self.viewer = viewer if viewer is not None else None
        self._label_choices = []
        self._intensity_choices = []
        self._p_sizes = None
        self._squeezed_dims = None
        self._prop = type('', (), {})()

        self._init_widgets()
        self._init_regionprops_container()
        self._init_layout()
        self._connect_events()
        
    def _init_widgets(self):
        self._image_directory = FileEdit(label="Image directory", mode="d")
        self._label_directory = FileEdit(label="Label directory", mode="d")
        self._output_directory = FileEdit(label="Output directory", mode="d")
        
        self._label_image = ComboBox(
            label="Label image",
            choices=self._label_choices,
            nullable=False,
            tooltip="Select label image to measure",
        )
        self._intensity_images = Select(
            label="Intensity images",
            choices=self._intensity_choices,
            allow_multiple=True,
            nullable=True,
            tooltip="Select intensity images to compare against labels",
        )
        self._scale_tuple = TupleEdit(
            value=(0.0000, 1.0000, 1.0000),
            label="Physical Pixel Sizes, ZYX",
            tooltip="Pixel size, usually in μm/px",
            options={"step": 0.0001},
        )
        self._measure_button = PushButton(label="Measure")
        
        self._progress_bar = ProgressBar(label="Progress:")
        # potentially add a textbox to insert a dict for regex expressions
        # potentially add a textbox to insert a dict for a treatment map
    def _init_regionprops_container(self):
        self._props_container = Container(layout="vertical")
        
        self._sk_props = [
            'area',
            'area_convex',
            'bbox',
            'centroid',
            'eccentricity',
            'extent', 
            'feret_diameter_max',
            'intensity_max',
            'intensity_mean',
            'intensity_min',
            'intensity_std',
            'num_pixels',
            'orientation',
            'perimeter',
            'solidity',
        ]
        
        for feature in self._sk_props:
            setattr(self._prop, feature, CheckBox(label=feature))
            self._props_container.extend(
                [getattr(self._prop, feature)]
            )
            
        self._prop.area.value = True
        self._prop.intensity_mean.value = True
    
    
    def _init_layout(self):
        self.extend([
            self._label_directory,
            self._image_directory,
            self._output_directory,
            self._label_image,
            self._intensity_images,
            self._scale_tuple,
            self._props_container,
            self._measure_button,
            self._progress_bar,
        ])
        
    def _connect_events(self):
        self._image_directory.changed.connect(self._update_image_choices)
        self._label_directory.changed.connect(self._update_label_choices)
        self._measure_button.clicked.connect(self.batch_measure)
    
    
    def _get_0th_img_from_dir(self, directory):
        from bioio import BioImage
        _, files = helpers.get_directory_and_files(directory)
        return BioImage(files[0])
    
    def _update_dim_and_scales(self, img):
        self._squeezed_dims = helpers.get_squeezed_dim_order(img)
        self._scale_tuple.value = (
            img.physical_pixel_sizes.Z or 1,
            img.physical_pixel_sizes.Y or 1,
            img.physical_pixel_sizes.X or 1,
        )
    
    def _update_image_choices(self):
        img = self._get_0th_img_from_dir(self._image_directory.value)
        img_channels = helpers.get_channel_names(img)
        img_channels = [f"Intensity: {channel}" for channel in img_channels]
        self._intensity_choices.extend(img_channels)
        self._intensity_images.choices = self._intensity_choices
        
    def _update_label_choices(self):
        img = self._get_0th_img_from_dir(self._label_directory.value)
        img_channels = helpers.get_channel_names(img)
        
        self._update_dim_and_scales(img)
        self._label_choices.extend(img_channels)
        self._label_image.choices = self._label_choices
        
        # add to the beginning of each string of the img_channels_list "Labels: "
        img_channels = [f"Labels: {channel}" for channel in img_channels]
        self._intensity_choices.extend(img_channels)
        self._intensity_images.choices = self._intensity_choices
        
        
    def batch_measure(self):
        from bioio import BioImage
        from skimage import measure
        # get all the files in the label directory
        label_dir, label_files = helpers.get_directory_and_files(self._label_directory.value)
        image_dir, image_files = helpers.get_directory_and_files(self._image_directory.value)

        # check if the label files are the same as the image files
        if len(label_files) != len(image_files):
            raise ValueError("Number of label files and image files do not match")
        
        log_loc = self._output_directory.value / ".log.txt"
        logger, handler = helpers.setup_logger(log_loc)

        logger.info(
            f"""
        Label Image: {self._label_image.value}
        Intensity Channels: {self._intensity_images.value}
        Num. Files: {len(label_files)}
        Image Directory: {image_dir}
        Label Directory: {label_dir}
        """
        )
        
        self._progress_bar.label = f"Measuring {len(label_files)} Images"
        self._progress_bar.value = 0
        self._progress_bar.max = len(label_files)
        
        measure_props_concat = []
        
        for idx, file in enumerate(label_files):
            logger.info(f"Processing file {file.name}")
            lbl = BioImage(label_dir / file.name)
            lbl_C = lbl.channel_names.index(self._label_image.value)
            label = lbl.get_image_data(self._squeezed_dims, C=lbl_C)
            
            intensity_images=[]
            
            # get the itnensity image only if the image directory is not empty
            if self._image_directory.value:
                image_path = image_dir / file.name
                if not image_path.exists():
                    logger.error(f"Image file {file.name} not found in intensity directory")
                    self._progress_bar.value = idx + 1
                    continue
                img = BioImage(image_path)
            # Get stack of intensity images
            if self._intensity_images.value:
                for channel in self._intensity_images.value:
                    if channel.startswith("Labels: "):
                        lbl_C = lbl.channel_names.index(channel[8:])
                        chan_img = lbl.get_image_data(self._squeezed_dims, C=lbl_C)
                    elif channel.startswith("Intensity: "):
                        img_C = img.channel_names.index(channel[11:])
                        chan_img = img.get_image_data(self._squeezed_dims, C=img_C)
                    intensity_images.append(chan_img)
                    
                intensity_stack = np.stack(intensity_images, axis=-1)
                
            else:
                intensity_stack = None
                    
                    
            # get the relevant spacing for regionprops, depending on length
            props_scale = self._scale_tuple.value
            props_scale = props_scale[-len(self._squeezed_dims):]
            # get the properties list
            # TODO: this is returning the widget class, not the name of the attribute
            properties = [prop.label for prop in self._props_container if prop.value]
            print(properties)
            # start the measuring here
            measure_props = measure.regionprops_table(
                label_image=label,
                intensity_image=intensity_stack,
                properties=properties, 
                spacing=props_scale,
            )
            
            measure_props_df = pd.DataFrame(measure_props)
            measure_props_df.insert(0, 'file', file.stem)
            
            measure_props_concat.append(measure_props_df)
            
        measure_props_concat = pd.concat(measure_props_concat)
        measure_props_concat.to_csv(self._output_directory.value / "measure_props.csv")