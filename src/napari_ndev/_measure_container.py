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

        self._init_widgets()
        self._init_layout()
        self._connect_events()
        
    def _init_widgets(self):
        self._image_directory = FileEdit(label="Image directory", mode="d")
        self._label_directory = FileEdit(label="Label directory", mode="d")
        self._output_directory = FileEdit(label="Output directory", mode="d")
        
        self._label_image = ComboBox(
            label="Label image",
            choices=self._label_choices,
            nullable=True,
            tooltip="Select label image to measure",
        )
        self._intensity_images = Select(
            label="Intensity images",
            choices=self._intensity_choices,
            allow_multiple=True,
            nullable=True,
            tooltip="Select intensity images to compare against labels",
        )
        self._progress_bar = ProgressBar(label="Progress:")
        # potentially add a textbox to insert a dict for regex expressions
        # potentially add a textbox to insert a dict for a treatment map
        
    def _init_layout(self):
        self.extend([
            self._label_directory,
            self._image_directory,
            self._output_directory,
            self._label_image,
            self._intensity_images,
            self._progress_bar,
        ])
        
    def _connect_events(self):
        self._image_directory.changed.connect(self._update_image_choices)
        self._label_directory.changed.connect(self._update_label_choices)
        
    def _update_metadata_from_file(self, file):
        from bioio import BioImage
        img = BioImage(file)
        return helpers.get_channel_names(img)
    
    def _update_image_choices(self):
        _, files = helpers.get_directory_and_files(self._image_directory.value)
        img_channels = self._update_metadata_from_file(files[0])
        img_channels = [f"Intensity: {channel}" for channel in img_channels]
        self._intensity_choices.extend(img_channels)
        self._intensity_images.choices = self._intensity_choices
        
    def _update_label_choices(self):
        _, files = helpers.get_directory_and_files(self._label_directory.value)
        img_channels = self._update_metadata_from_file(files[0])
        self._label_choices.extend(img_channels)
        self._label_image.choices = self._label_choices
        
        # add to the beginning of each string of the img_channels_list "Labels: "
        img_channels = [f"Labels: {channel}" for channel in img_channels]
        self._intensity_choices.extend(img_channels)
        self._intensity_images.choices = self._intensity_choices
        
        
    def batch_measure(self):
        from bioio import BioImage
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
        Intensity Channels: {self._intensity_choices.value}
        Num. Files: {len(label_files)}
        Image Directory: {image_dir}
        Label Directory: {label_dir}
        """
        )
        
        self._progress_bar.label = f"Measuring {len(label_files)} Images"
        self._progress_bar.value = 0
        self._progress_bar.max = len(label_files)
        
        for idx, file in enumerate(label_files):
            logger.info(f"Processing file {file.name}")
            lbl = BioImage(label_dir / file.name)
            lbl_C = lbl.channel_names.index(self._label_image.value)
            label = lbl.get_image_data("TCZYX", C=lbl_C)
            
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
                        chan_img = lbl.get_image_data("TCZYX", C=lbl_C)
                    elif channel.startswith("Intensity: "):
                        img_C = img.channel_names.index(channel[11:])
                        chan_img = img.get_image_data("TCZYX", C=img_C)
                    intensity_images.append(chan_img)
            else:
                intensity_images = None
                    
            # start the measuring here