from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    LineEdit,
    ProgressBar,
    PushButton,
    Select,
    TextEdit,
    TupleEdit,
)
from qtpy.QtWidgets import QTabWidget

from napari_ndev import helpers

if TYPE_CHECKING:
    import pathlib

    from bioio import BioImage

    import napari


class MeasureContainer(Container):
    """Widget to measure labels from folders.

    This class provides functionality to measure labels and compare them against intensity images, which can be microscopic images or other labels. It initializes various widgets and containers for user input and interaction, and connects events to handle user actions.

    Parameters
    ----------
    viewer : napari.viewer.Viewer
        The napari viewer instance. Optional.

    Attributes
    ----------
    viewer : napari.viewer.Viewer
        The napari viewer instance.

    _label_choices : list
        List of label choices.
    _intensity_choices : list
        List of intensity image choices.
    _p_sizes : None
        Placeholder for pixel sizes.
    _squeezed_dims : None
        Placeholder for squeezed dimensions.
    _prop : object
        Dynamic object to hold region properties checkboxes.
    _label_directory : FileEdit
        Widget for selecting label directory.
    _image_directory : FileEdit
        Widget for selecting image directory.
    _region_directory : FileEdit
        Widget for selecting region directory.
    _output_directory : FileEdit
        Widget for selecting output directory.
    _label_image : ComboBox
        Widget for selecting label image.
    _intensity_images : Select
        Widget for selecting intensity images.
    _scale_tuple : TupleEdit
        Widget for setting physical pixel sizes.
    _measure_button : PushButton
        Button to start measurement.
    _progress_bar : ProgressBar
        Progress bar to show measurement progress.
    _props_container : Container
        Container for region properties checkboxes.
    _sk_props : list
        List of region properties.
    _id_regex_container : Container
        Container for ID regex settings.
    _example_id_string : LineEdit
        Widget for example ID string.
    _id_regex_dict : TextEdit
        Widget for ID regex dictionary.
    _tx_map_container : Container
        Container for treatment map settings.
    _tx_id : LineEdit
        Widget for treatment ID.
    _tx_n_well : ComboBox
        Widget for number of wells.
    _tx_dict : TextEdit
        Widget for treatment dictionary.
    _grouping_container : Container
        Container for grouping settings.
    _create_grouped : CheckBox
        Checkbox to create grouped data.
    _group_by_sample_id : CheckBox
        Checkbox to group by sample ID.

    Methods
    -------
    _init_widgets()
        Initializes the widgets for user input.
    _init_regionprops_container()
        Initializes the container for region properties checkboxes.
    _init_id_regex_container()
        Initializes the container for ID regex settings.
    _init_tx_map_container()
        Initializes the container for treatment map settings.
    _init_grouping_container()
        Initializes the container for grouping settings.
    _init_layout()
        Initializes the layout of the container.
    _connect_events()
        Connects events to handle user actions.
    _get_0th_img_from_dir(directory)
        Gets the first image from a directory.
    _update_dim_and_scales(img)
        Updates the dimensions and scales based on the image.
    _update_choices(directory, prefix, update_label=False)
        Updates the choices for labels and intensity images.
    _update_image_choices()
        Updates the choices for intensity images.
    _update_label_choices()
        Updates the choices for label images.
    _update_region_choices()
        Updates the choices for region images.
    _safe_dict_eval(dict_string, dict_name=None)
        Safely evaluates a dictionary string.
    batch_measure()
        Performs batch measurement of labels and intensity images, and returns the measurement results as a DataFrame.

    """

    def __init__(
        self,
        viewer: napari.viewer.Viewer = None,
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
        self._init_id_regex_container()
        self._init_tx_map_container()
        self._init_grouping_container()
        self._init_layout()
        self._connect_events()

    def _init_widgets(self):
        self._label_directory = FileEdit(label='Label directory', mode='d')
        self._image_directory = FileEdit(
            label='Image directory', mode='d', nullable=True
        )
        self._region_directory = FileEdit(
            label='Region directory', mode='d', nullable=True
        )
        self._output_directory = FileEdit(label='Output directory', mode='d')

        self._label_image = ComboBox(
            label='Label image',
            choices=self._label_choices,
            nullable=False,
            tooltip='Select label image to measure',
        )
        self._intensity_images = Select(
            label='Intensity images',
            choices=self._intensity_choices,
            allow_multiple=True,
            nullable=True,
            tooltip='Select intensity images to compare against labels',
        )
        self._scale_tuple = TupleEdit(
            value=(0.0000, 1.0000, 1.0000),
            label='Physical Pixel Sizes, ZYX',
            tooltip='Pixel size, usually in μm/px',
            options={'step': 0.0001},
        )
        self._measure_button = PushButton(label='Measure')

        self._progress_bar = ProgressBar(label='Progress:')

    def _init_regionprops_container(self):
        self._props_container = Container(layout='vertical')

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
            self._props_container.extend([getattr(self._prop, feature)])

        self._prop.area.value = True

    def _init_id_regex_container(self):
        self._id_regex_container = Container(layout='vertical')
        self._example_id_string = LineEdit(
            label='Example ID String',
            value=None,
            nullable=True,
        )
        self._id_regex_dict = TextEdit(
            label='ID Regex Dict',
            value='{\n\n}',
        )
        self._id_regex_container.extend(
            [self._example_id_string, self._id_regex_dict]
        )

    def _init_tx_map_container(self):
        self._tx_map_container = Container(layout='vertical')
        self._tx_id = LineEdit(
            label='Treatment ID',
            value=None,
            nullable=True,
            tooltip='Usually, the treatment ID is the well ID or a unique identifier for each sample'
            "The treatment dict will be looked up against whatever this value is. If it is 'file', then will match against the filename",
        )
        self._tx_n_well = ComboBox(
            label='Number of Wells',
            value=None,
            choices=[6, 12, 24, 48, 96, 384],
            nullable=True,
            tooltip='By default, treatments must be verbosely defined for each condition and sample id'
            'If you have a known plate map, then selecting wells will allow a sparse treatment map to be passed to PlateMapper',
        )
        self._tx_dict = TextEdit(label='Treatment Dict', value='{\n\n}')
        # TODO: Add example treatment regex result widget when example id string or id regex dict is changed

        self._tx_map_container.extend(
            [self._tx_id, self._tx_n_well, self._tx_dict]
        )

    def _init_grouping_container(self):
        self._grouping_container = Container(layout='vertical')
        self._create_grouped = CheckBox(
            label='Create Grouped Data',
            value=False,
            tooltip='If checked, will create a grouped data frame with the same properties as the original data frame',
        )
        self._group_by_sample_id = CheckBox(
            label='Group by Sample ID',
            value=True,
            tooltip='If checked, will group the data by the id_string, which is usually the filename and scene',
        )

        self._grouping_container.extend(
            [self._create_grouped, self._group_by_sample_id]
        )

    def _init_layout(self):
        self.extend(
            [
                self._label_directory,
                self._image_directory,
                self._region_directory,
                self._output_directory,
                self._label_image,
                self._intensity_images,
                self._scale_tuple,
                self._measure_button,
                self._progress_bar,
            ]
        )

        tabs = QTabWidget()
        tabs.addTab(self._props_container.native, 'Region Props')
        tabs.addTab(self._id_regex_container.native, 'ID Regex')
        tabs.addTab(self._tx_map_container.native, 'Tx Map')
        tabs.addTab(self._grouping_container.native, 'Grouping')
        self.native.layout().addWidget(tabs)

    def _connect_events(self):
        self._image_directory.changed.connect(self._update_image_choices)
        self._label_directory.changed.connect(self._update_label_choices)
        self._region_directory.changed.connect(self._update_region_choices)
        self._measure_button.clicked.connect(self.batch_measure)

    def _get_0th_img_from_dir(
        self, directory: str | None = None
    ) -> tuple[BioImage, pathlib.Path]:
        from bioio import BioImage

        _, files = helpers.get_directory_and_files(directory)
        return BioImage(files[0]), files[0]

    def _update_dim_and_scales(self, img):
        self._squeezed_dims = helpers.get_squeezed_dim_order(img)
        self._scale_tuple.value = (
            img.physical_pixel_sizes.Z or 1,
            img.physical_pixel_sizes.Y or 1,
            img.physical_pixel_sizes.X or 1,
        )

    def _update_choices(self, directory, prefix, update_label=False):
        img, _ = self._get_0th_img_from_dir(directory)
        img_channels = helpers.get_channel_names(img)
        img_channels = [f'{prefix}: {channel}' for channel in img_channels]

        if update_label:
            self._update_dim_and_scales(img)
            self._label_choices.extend(img_channels)
            self._label_image.choices = self._label_choices

        self._intensity_choices.extend(img_channels)
        self._intensity_images.choices = self._intensity_choices

    def _update_image_choices(self):
        self._update_choices(self._image_directory.value, 'Intensity')

    def _update_label_choices(self):
        self._update_choices(
            self._label_directory.value, 'Labels', update_label=True
        )
        img, file_id = self._get_0th_img_from_dir(self._label_directory.value)
        id_string = helpers.create_id_string(img, file_id.stem)
        self._example_id_string.value = id_string

    def _update_region_choices(self):
        self._update_choices(self._region_directory.value, 'Region')

    def _safe_dict_eval(self, dict_string, dict_name=None):
        if dict_string is None:
            return None

        stripped_string = dict_string.strip()
        if stripped_string == '{}' or not stripped_string:
            return None
        try:
            return ast.literal_eval(stripped_string)
        except (ValueError, SyntaxError):
            return None

    def batch_measure(self) -> pd.DataFrame:
        from bioio import BioImage

        # from skimage import measure
        from napari_ndev import measure as ndev_measure

        # get all the files in the label directory
        label_dir, label_files = helpers.get_directory_and_files(
            self._label_directory.value
        )
        image_dir, image_files = helpers.get_directory_and_files(
            self._image_directory.value
        )
        region_dir, region_files = helpers.get_directory_and_files(
            self._region_directory.value
        )

        # check if the label files are the same as the image files
        # if self._image_directory.value is not None:
        #     if len(label_files) != len(image_files):
        #         raise ValueError(
        #             'Number of label files and image files do not match'
        #         )
        # if self._region_directory.value is not None:
        #     if len(label_files) != len(region_files):
        #         raise ValueError(
        #             'Number of label files and region files do not match'
        #         )

        log_loc = self._output_directory.value / '.log.txt'
        logger, handler = helpers.setup_logger(log_loc)

        logger.info(
            """
            Label Image: %s
            Intensity Channels: %s
            Num. Files: %d
            Label Directory: %s
            Image Directory: %s
            Region Directory: %s
            Output Directory: %s
            ID Example: %s
            ID Regex Dict: %s
            Tx ID: %s
            Tx N Well: %s
            Tx Dict: %s
            """,
            self._label_image.value,
            self._intensity_images.value,
            len(label_files),
            label_dir,
            image_dir,
            region_dir,
            self._output_directory.value,
            self._example_id_string.value,
            self._id_regex_dict.value,
            self._tx_id.value,
            self._tx_n_well.value,
            self._tx_dict.value
        )

        self._progress_bar.label = f'Measuring {len(label_files)} Images'
        self._progress_bar.value = 0
        self._progress_bar.max = len(label_files)
        # get the relevant spacing for regionprops, depending on length
        props_scale = self._scale_tuple.value
        props_scale = props_scale[-len(self._squeezed_dims) :]
        # get the properties list
        properties = [
            prop.label for prop in self._props_container if prop.value
        ]

        id_regex_dict = self._safe_dict_eval(
            self._id_regex_dict.value, 'ID Regex Dict'
        )
        tx_dict = self._safe_dict_eval(self._tx_dict.value, 'Tx Dict')
        measure_props_concat = []

        for idx, file in enumerate(label_files):
            # TODO: Add scene processing
            logger.info('Processing file %s', file.name)
            lbl = BioImage(label_dir / file.name)
            id_string = helpers.create_id_string(lbl, file.stem)

            label_chan = self._label_image.value[8:]
            lbl_C = lbl.channel_names.index(label_chan)

            intensity_images = []
            intensity_names = []

            # get the itnensity image only if the image directory is not empty
            if self._image_directory.value:
                image_path = image_dir / file.name
                if not image_path.exists():
                    logger.error(
                        'Image file %s not found in intensity directory', file.name
                    )
                    self._progress_bar.value = idx + 1
                    continue
                img = BioImage(image_path)
            if self._region_directory.value:
                region_path = region_dir / file.name
                if not region_path.exists():
                    logger.error(
                        'Region file %s not found in region directory', file.name
                    )
                    self._progress_bar.value = idx + 1
                    continue
                reg = BioImage(region_path)

            for scene_idx, _scene in enumerate(lbl.scenes):
                logger.info('Processing scene %s', scene_idx)
                lbl.set_scene(scene_idx)
                label = lbl.get_image_data(self._squeezed_dims, C=lbl_C)
                id_string = helpers.create_id_string(lbl, file.stem)

                # Get stack of intensity images if there are any selected
                if self._intensity_images.value and not None:
                    for channel in self._intensity_images.value:
                        if channel.startswith('Labels: '):
                            chan = channel[8:]
                            lbl_C = lbl.channel_names.index(chan)
                            lbl.set_scene(scene_idx)
                            chan_img = lbl.get_image_data(
                                self._squeezed_dims, C=lbl_C
                            )
                        elif channel.startswith('Intensity: '):
                            chan = channel[11:]
                            img_C = img.channel_names.index(chan)
                            img.set_scene(scene_idx)
                            chan_img = img.get_image_data(
                                self._squeezed_dims, C=img_C
                            )
                        elif channel.startswith('Region: '):
                            chan = channel[8:]
                            reg_C = reg.channel_names.index(chan)
                            img.set_scene(scene_idx)
                            chan_img = reg.get_image_data(
                                self._squeezed_dims, C=reg_C
                            )
                        intensity_names.append(chan)
                        intensity_images.append(chan_img)

                    # the last dim is the multi-channel dim for regionprops
                    intensity_stack = np.stack(intensity_images, axis=-1)

                else:
                    intensity_stack = None
                    intensity_names = None

                # start the measuring here
                # TODO: Add optional scaling, in case images have different scales?
                measure_props_df = ndev_measure.measure_regionprops(
                    label_images=label,
                    label_names=label_chan,
                    intensity_images=intensity_stack,
                    intensity_names=intensity_names,
                    properties=properties,
                    scale=props_scale,
                    id_string=id_string,
                    id_regex_dict=id_regex_dict,
                    tx_id=self._tx_id.value,
                    tx_dict=tx_dict,
                    tx_n_well=self._tx_n_well.value,
                    save_data_path=None,
                )

                measure_props_concat.append(measure_props_df)
                self._progress_bar.value = idx + 1

        measure_props_df = pd.concat(measure_props_concat)
        measure_props_df.to_csv(
            self._output_directory.value / f'measure_props_{label_chan}.csv'
        )

        if self._create_grouped.value:
            if self._group_by_sample_id.value:
                # get count data
                measure_props_count = (
                    measure_props_df.groupby('id')
                    .agg({measure_props_df.columns[0]: 'count'})
                    .rename(
                        columns={measure_props_df.columns[0]: 'label_count'}
                    )
                    .reset_index(drop=True)
                )
                measure_props_grouped = (
                    measure_props_df.groupby('id')  # sw
                    .agg(
                        {
                            col: ['mean', 'std']
                            for col in measure_props_df.columns[1:]
                        }
                    )
                    .reset_index()
                )  # genereates a multi-index
                # collapse multi index and combine columns names with '_' sep
                measure_props_grouped.columns = [
                    f'{col[0]}_{col[1]}' if col[1] else col[0]
                    for col in measure_props_grouped.columns
                ]

                measure_props_grouped = pd.concat(
                    [measure_props_grouped, measure_props_count], axis=1
                )

            measure_props_grouped.to_csv(
                self._output_directory.value
                / f'measure_props_grouped_{label_chan}.csv'
            )
        else:
            measure_props_grouped = None

        logger.removeHandler(handler)

        return measure_props_df, measure_props_grouped