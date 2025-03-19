from __future__ import annotations

import ast
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from magicclass.widgets import (
    GroupBoxContainer,
    ScrollableContainer,
)
from magicgui.widgets import (
    CheckBox,
    Container,
    FileEdit,
    Label,
    LineEdit,
    PushButton,
    TextEdit,
    TupleEdit,
)

from napari.layers import (
    Image as ImageLayer,
    Labels as LabelsLayer,
    Shapes as ShapesLayer,
)
from napari_ndev import get_settings, helpers, nImage

if TYPE_CHECKING:
    from bioio import BioImage

    import napari
    from napari.layers import Layer


class UtilitiesContainer(ScrollableContainer):
    """
    A widget to work with images and labels in the napari viewer.

    Parameters
    ----------
    viewer: napari.viewer.Viewer, optional
        The napari viewer instance.

    Attributes
    ----------
    _viewer: napari.viewer.Viewer
        The napari viewer instance.
    _image_save_dims: str or None
        The dimension order for saving images.
    _label_save_dims: str or None
        The dimension order for saving labels.
    _p_sizes: PhysicalPixelSizes
        The physical pixel sizes for the image.
    _files: FileEdit
        Widget for selecting file(s).
    _open_image_button: PushButton
        Button for opening images.
    _save_directory: FileEdit
        Widget for selecting the save directory.
    _save_name: LineEdit
        Widget for entering the file save name.
    _metadata_from_selected_layer: PushButton
        Button for updating metadata from the selected layer.
    _dim_order: LineEdit
        Widget for entering the dimension order.
    _channel_names: LineEdit
        Widget for entering the channel names.
    _physical_pixel_sizes_z: FloatSpinBox
        Widget for entering the Z pixel size in micrometers.
    _physical_pixel_sizes_y: FloatSpinBox
        Widget for entering the Y pixel size in micrometers.
    _physical_pixel_sizes_x: FloatSpinBox
        Widget for entering the X pixel size in micrometers.
    _image_layer: Select
        Widget for selecting the image layer.
    _concatenate_image_files: CheckBox
        Checkbox for concatenating image files.
    _concatenate_image_layers: CheckBox
        Checkbox for concatenating image layers.
    _save_image_button: PushButton
        Button for saving images.
    _labels_layer: Widget
        Widget for working with labels layer.
    _save_labels_button: PushButton
        Button for saving labels.
    _shapes_layer: Widget
        Widget for working with shapes layer.
    _save_shapes_button: PushButton
        Button for saving shapes as labels.
    _results: TextEdit
        Widget for displaying information.

    Methods
    -------
    _update_metadata(img)
        Update the metadata based on the given image.
    update_metadata_from_file()
        Update the metadata from the selected file.
    update_metadata_from_layer()
        Update the metadata from the selected layer.
    open_images()
        Open the selected images in the napari viewer.
    concatenate_images(concatenate_files, files, concatenate_layers, layers)
        Concatenate the image data based on the selected options.
    p_sizes()
        Get the physical pixel sizes.
    _get_save_loc(parent)
        Get the save location based on the parent directory.
    _common_save_logic(data, uri, dim_order, channel_names, layer)
        Common logic for saving data as OME-TIFF.
    save_ome_tiff()
        Save the concatenated image data as OME-TIFF.
    save_labels()
        Save the labels data.
    save_shapes_as_labels()
        Save the shapes data as labels.

    """

    def __init__(self, viewer: napari.viewer.Viewer = None):
        """
        Initialize the UtilitiesContainer widget.

        Parameters
        ----------
        viewer : napari.viewer.Viewer, optional
            The napari viewer instance.

        """
        super().__init__(labels=False)

        self.min_width = 500 # TODO: remove this hardcoded value
        self._viewer = viewer if viewer is not None else None
        self._squeezed_dims_order: str | None = None
        self._squeezed_dims: tuple[int, ...] | None = None
        self._settings = get_settings()

        self._init_widgets()
        self._init_save_name_container()
        self._init_open_image_container()
        self._init_metadata_container()
        self._init_concatenate_files_container()
        self._init_save_layers_container()
        self._init_scene_container()
        # self._init_figure_options_container() # TODO: add figure saving
        self._init_layout()
        self._connect_events()

    def _init_widgets(self):
        """Initialize widgets."""
        self._save_directory_prefix = LineEdit(
            label='Save Dir. Prefix',
            tooltip='Prefix for the save directories.',
        )
        self._save_directory = FileEdit(
            mode='d',
            tooltip='Directory where images will be saved. \n'
            'Upon selecting the first file, the save directory will be set \n'
            'to the grandparent directory of the first file.',
        )
        self._save_directory_container = Container(
            widgets=[self._save_directory_prefix, self._save_directory],
            layout='horizontal',
        )
        self._default_save_directory = self._save_directory.value
        self._files = FileEdit(
            mode='rm',
            tooltip='Select file(s) to load.',
        )

        self._results = TextEdit(label='Info')

    def _init_save_name_container(self):
        """Initialize the save name container."""
        self._save_name_container = Container(layout='horizontal')
        self._save_name = LineEdit(
            label='Save Name',
            tooltip='Name of the saved file. '
            'Proper extension will be added when saved.',
        )
        self._append_scene_button = PushButton(
            label='Append Scene to Name',
        )
        self._save_name_container.extend([
            self._save_name,
            self._append_scene_button
        ])

    def _init_open_image_container(self):
        """Initialize the open image container."""
        self._open_image_container = Container(layout='horizontal')
        self._open_image_button = PushButton(label='Open File(s)')
        self._select_next_image_button = PushButton(
            label='Select Next',
            tooltip='Select the next file(s) in the directory. \n'
            'Note that the files are sorted alphabetically and numerically.'
        )
        self._open_image_container.append(self._open_image_button)
        self._open_image_container.append(self._select_next_image_button)

    def _init_concatenate_files_container(self):
        self._concatenate_files_container = Container(
            layout='horizontal',
        )
        self._concatenate_files_button = PushButton(label='Concat. Files')
        self._concatenate_batch_button = PushButton(
            label='Batch Concat.',
            tooltip='Concatenate files in the selected directory by iterating'
            ' over the remaing files in the directory based on the number of'
            ' files selected. The files are sorted '
            'alphabetically and numerically, which may not be consistent '
            'with your file viewer. But, opening related consecutive files '
            'should work as expected.',
        )
        self._concatenate_files_container.extend([
            self._concatenate_files_button,
            self._concatenate_batch_button,
        ])


    def _init_metadata_container(self):
        self._update_scale = CheckBox(
            value=True,
            label='Scale',
            tooltip='Update the scale when files are selected.',
        )
        self._update_channel_names = CheckBox(
            value=True,
            label='Channel Names',
            tooltip='Update the channel names when files are selected.',
        )
        self._file_options_container = GroupBoxContainer(
            layout='horizontal',
            name='Update Metadata on File Select',
            labels=False,
            label=False,
            widgets=[self._update_scale, self._update_channel_names]
        )

        self._layer_metadata_update_button = PushButton(
            label='Update from Selected Layer'
        )
        self._num_scenes_label = Label(
            label='Num. Scenes: ',
        )
        self._dim_shape = LineEdit(
            label='Dims: ',
            tooltip='Sanity check for available dimensions.',
        )
        self._image_info_container = Container(
            widgets=[self._num_scenes_label, self._dim_shape],
            layout='horizontal',
        )

        self._channel_names = LineEdit(
            label='Channel Name(s)',
            tooltip='Enter channel names as a list. If left blank or the '
            'channel names are not the proper length, then default channel '
            'names will be used.',
        )

        self._scale_tuple = TupleEdit(
            label='Scale, ZYX',
            tooltip='Pixel size, usually in μm',
            value=(0.0000, 1.0000, 1.0000),
            options={'step': 0.0001},
        )
        self._channel_scale_container = Container(
            widgets=[self._channel_names, self._scale_tuple],
        )
        self._scale_layers_button = PushButton(
            label='Scale Layer(s)',
            tooltip='Scale the selected layer(s) based on the given scale.',
        )
        self._metadata_button_container = Container(
            widgets=[
                self._layer_metadata_update_button,
                self._scale_layers_button
            ],
            layout='horizontal',
        )

        self._metadata_container = GroupBoxContainer(
            layout='vertical',  # label='Update Metadata from',
            name='Metadata',
            widgets=[
                self._file_options_container,
                self._image_info_container,
                self._channel_scale_container,
                self._metadata_button_container,
            ],
            labels=False,
        )

    def _init_scene_container(self):
        """Initialize the scene container, allowing scene saving."""
        self._scene_container = Container(
            layout='horizontal',
            tooltip='Must be in list index format. Ex: [0, 1, 2] or [5:10]',
        )
        self._scenes_to_extract = LineEdit(
            # label="Scenes to Extract",
            tooltip='Enter the scenes to extract as a list. If left blank '
            'then all scenes will be extracted.',
        )
        self._extract_scenes = PushButton(
            label='Extract and Save Scenes',
            tooltip='Extract scenes from a single selected file.',
        )
        self._scene_container.append(self._scenes_to_extract)
        self._scene_container.append(self._extract_scenes)

    def _init_save_layers_container(self):
        """Initialize the container to save images, labels, and shapes."""
        self._save_layers_button = PushButton(
            text='Selected Layers (TIFF)',
            tooltip='Concatenate and save all selected layers as OME-TIFF. '
            'Layers will save to corresponding directories based on the layer '
            'type, e.g. Images, Labels, ShapesAsLabels. Shapes are saved as '
            'labels based on the selected image layer dimensions. If multiple '
            'layer types are selected, then the image will save to Layers.',
        )
        self._export_figure_button = PushButton(
            text='Figure (PNG)',
            tooltip='Export the current canvas figure as a PNG to the Figure '
            'directory. Only works in 2D mode. Use Screenshot for 3D figures. '
            'Crops the figure to the extent of the data, attempting to remove '
            'margins. Increase or decrease scaling in the settings',
        )
        self._export_screenshot_button = PushButton(
            text='Canvas (PNG)',
            tooltip='Export the current canvas screenshot as a PNG to the '
            'Figure directory. Works in 2D and 3D mode. Uses the full canvas '
            'size, including margins. Increase or decrease scaling in the '
            'settings, and also it is possible to override the canvas size.',
        )

        self._save_layers_container = GroupBoxContainer(
            layout='horizontal',
            name='Export',
            labels=None,
        )

        self._save_layers_container.extend([
            self._save_layers_button,
            self._export_figure_button,
            self._export_screenshot_button,
        ])

    def _init_layout(self):
        """Initialize the layout of the widget."""
        self._file_group = GroupBoxContainer(
            widgets=[
                self._files,
                self._open_image_container,
            ],
            name='Opening',
            labels=False,
        )
        self._save_group = GroupBoxContainer(
            widgets=[
                self._save_directory_container,
                self._save_name_container,
                self._concatenate_files_container,
                self._scene_container,
                self._save_layers_container,
            ],
            name='Saving',
            labels=False,
        )

        self.extend(
            [
                self._file_group,
                self._save_group,
                self._metadata_container,
                self._results,
            ]
        )

    def _connect_events(self):
        """Connect the events of the widgets to respective methods."""
        self._files.changed.connect(self.update_save_directory)
        self._files.changed.connect(self.update_metadata_on_file_select)
        self._append_scene_button.clicked.connect(self.append_scene_to_name)
        self._open_image_button.clicked.connect(self.open_images)
        self._select_next_image_button.clicked.connect(self.select_next_images)

        self._layer_metadata_update_button.clicked.connect(
            self.update_metadata_from_layer
        )
        self._scale_layers_button.clicked.connect(self.rescale_by)

        self._concatenate_files_button.clicked.connect(self.save_files_as_ome_tiff)
        self._concatenate_batch_button.clicked.connect(self.batch_concatenate_files)
        self._extract_scenes.clicked.connect(self.save_scenes_ome_tiff)
        self._save_layers_button.clicked.connect(self.save_layers_as_ome_tiff)
        self._export_figure_button.clicked.connect(self.canvas_export_figure)
        self._export_screenshot_button.clicked.connect(self.canvas_screenshot)
        self._results._on_value_change()

    @property
    def p_sizes(self):
        """
        Get the physical pixel sizes.

        Returns
        -------
        PhysicalPixelSizes
            The physical pixel sizes.

        """
        from bioio_base.types import PhysicalPixelSizes

        return PhysicalPixelSizes(
            self._scale_tuple.value[0],
            self._scale_tuple.value[1],
            self._scale_tuple.value[2],
        )

    def update_save_directory(self):
        """Update the save directory based on the selected files."""
        if self._save_directory.value == self._default_save_directory:
            self._save_directory.value = self._files.value[0].parent.parent

    def _update_metadata_from_Image(
        self,
        img: BioImage,
        update_channel_names: bool = True,
        update_scale: bool = True,
    ):
        """
        Update the metadata based on the given image.

        Parameters
        ----------
        img : BioImage
            The image from which to update the metadata.
        update_channel_names : bool, optional
            Update the channel names, by default True.
        update_scale : bool, optional
            Update the scale, by default True.

        """
        # get the part of the string between the brackets, which is the dim order
        dims = re.search(r'\[(.*?)\]', str(img.dims)).group(1)
        self._dim_shape.value = dims
        self._num_scenes_label.value = str(len(img.scenes))

        self._squeezed_dims_order = helpers.get_squeezed_dim_order(img)
        dims_tuple = tuple(d for d in self._squeezed_dims_order)
        self._squeezed_dims = img.dims[dims_tuple]

        if update_channel_names:
            self._channel_names.value = helpers.get_channel_names(img)
        if update_scale:
            self._scale_tuple.value = (
                img.physical_pixel_sizes.Z or 1,
                img.physical_pixel_sizes.Y or 1,
                img.physical_pixel_sizes.X or 1,
            )

    def update_metadata_on_file_select(self):
        """Update self._save_name.value and metadata if selected."""
        # TODO: get true stem of file, in case .ome.tiff
        self._save_name.value = str(self._files.value[0].stem)
        img = nImage(self._files.value[0])

        self._update_metadata_from_Image(
            img,
            update_channel_names=self._update_channel_names.value,
            update_scale=self._update_scale.value,
        )

    def append_scene_to_name(self):
        """Append the scene to the save name."""
        if self._viewer.layers.selection.active is not None:
            try:
                img = self._viewer.layers.selection.active.metadata['bioimage']
                # remove bad characters from scene name
                scene = re.sub(r'[^\w\s]', '-', img.current_scene)
                self._save_name.value = f'{self._save_name.value}_{scene}'
            except AttributeError:
                self._results.value = (
                    'Tried to append scene to name, but layer not opened with'
                    ' nDev reader.'
                )
        else:
            self._results.value = (
                'Tried to append scene to name, but no layer selected.'
                ' So the first scene from the first file will be appended.'
            )
            img = nImage(self._files.value[0])
            scene = re.sub(r'[^\w\s]', '-', img.current_scene)
            self._save_name.value = f'{self._save_name.value}_{scene}'

    def update_metadata_from_layer(self):
        """
        Update metadata from the selected layer.

        Expects images to be opened with napari-ndev reader.

        Note:
        ----
        This should also support napari-bioio in the future, when released.

        """
        selected_layer = self._viewer.layers.selection.active
        try:
            img = selected_layer.metadata['bioimage']
            self._update_metadata_from_Image(img)

        except AttributeError:
            self._results.value = (
                'Tried to update metadata, but no layer selected.'
                f'\nAt {time.strftime("%H:%M:%S")}'
            )
        except KeyError:
            scale = selected_layer.scale
            self._scale_tuple.value = (
                scale[-3] if len(scale) >= 3 else 1,
                scale[-2],
                scale[-1],
            )
            self._results.value = (
                'Tried to update metadata, but could only update scale'
                ' because layer not opened with nDev reader.'
                f'\nAt {time.strftime("%H:%M:%S")}'
            )

    def open_images(self):
        """Open the selected images in the napari viewer with napari-ndev."""
        self._viewer.open(self._files.value, plugin='napari-ndev')

    # Converted
    def select_next_images(self):
        from natsort import os_sorted
        """Open the next set of images in the directyory."""
        num_files = self._files.value.__len__()

        # get the parent directory of the first file
        first_file = self._files.value[0]
        parent_dir = first_file.parent

        # get the list of files in the parent directory
        files = list(parent_dir.glob(f'*{first_file.suffix}'))
        # sort the files naturally (case-insensitive and numbers in order)
        # like would be scene in windows file explorer default sorting
        # https://pypi.org/project/natsort/#sort-paths-like-my-file-browser-e-g-windows-explorer-on-windows

        files = os_sorted(files)

        # get the index of the first file in the list and then the next files
        idx = files.index(first_file)
        next_files = files[idx + num_files : idx + num_files + num_files]

        # if there are no more files, then return
        if not next_files:
            self._results.value = (
                'No more file sets to select.'
            )
            return
        # set the nwe save names, and update the file value
        img = nImage(next_files[0])

        self._save_name.value = helpers.create_id_string(img, next_files[0].stem)
        self._files.value = next_files

        self.update_metadata_on_file_select()

    def rescale_by(self):
        """Rescale the selected layers based on the given scale."""
        layers = self._viewer.layers.selection
        scale_tup = self._scale_tuple.value

        for layer in layers:
            scale_len = len(layer.scale)
            # get the scale_tup from the back of the tuple first, in case dims
            # are missing in the new layer
            layer.scale = scale_tup[-scale_len:]

    @staticmethod
    def concatenate_files(
        files: str | Path | list[str | Path],
    ) -> np.ndarray:
        """
        Concatenate the image data from the selected files.

        Removes "empty" channels, which are channels with no values above 0.
        This is present in some microscope formats where it will image in RGB,
        and then leave empty channels not represented by the color channels.

        Does not currently handle scenes.

        Parameters
        ----------
        files : str or Path or list of str or Path
            The file(s) to concatenate.

        Returns
        -------
        numpy.ndarray
            The concatenated image data.

        """
        array_list = []

        for file in files:
            img = nImage(file)

            if 'S' in img.dims.order:
                img_data = img.get_image_data('TSZYX')
            else:
                img_data = img.data

            # iterate over all channels and only keep if not blank
            for idx in range(img_data.shape[1]):
                array = img_data[:, [idx], :, :, :]
                if array.max() > 0:
                    array_list.append(array)
        return np.concatenate(array_list, axis=1)

    def concatenate_layers(
        self,
        layers: Layer | list[Layer],
    ) -> np.ndarray:
        """
        Concatenate the image data from the selected layers.

        Adapts all layers to 5D arrays for compatibility with image dims.
        If the layer is a shapes layer, it will look for a corresponding image
        layer to get the dimensions for the shapes layer.

        Parameters
        ----------
        layers : napari.layers.Image or list of napari.layers.Image
            The selected image layers.

        Returns
        -------
        numpy.ndarray
            The concatenated image data.

        """
        if any(isinstance(layer, ShapesLayer) for layer in layers):
            shape_to_label_dim = self._get_dims_for_shape_layer()

        array_list = []

        for layer in layers:
            if isinstance(layer, ShapesLayer):
                layer_data = layer.to_labels(labels_shape=shape_to_label_dim)
                layer_data = layer_data.astype(np.int16)
            else:
                layer_data = layer.data

            array_list.append(layer_data)

        # stack along a new axis, which will be the channel axis for saving
        return np.stack(array_list, axis=0)

    def _get_dims_for_shape_layer(self) -> tuple[int, ...]:
        # use dimension of the current image file
        if self._squeezed_dims is not None:
            return self._squeezed_dims

        # if no dims available, get it from the first instance in the viewer
        dim_layer = next(
                (layer for layer in self._viewer.layers if isinstance(layer, (ImageLayer, LabelsLayer))),
                None,
            )
        if dim_layer is None:
            raise ValueError('No image or labels present to convert shapes layer.')
        label_dim = dim_layer.data.shape
            # drop last axis if represents RGB image
        label_dim = label_dim[:-1] if label_dim[-1] == 3 else label_dim

        return label_dim

    def _get_save_loc(
        self, root_dir: Path, parent: str, file_name: str
    ) -> Path:
        """
        Get the save location based on the parent directory.

        Parameters
        ----------
        root_dir : Path
            The root directory.
        parent : str
            The parent directory. eg. 'Image', 'Labels', 'ShapesAsLabels'
        file_name : str
            The file name.

        Returns
        -------
        Path
            The save location. root_dir / parent / file_name

        """
        save_directory = root_dir / parent
        save_directory.mkdir(parents=False, exist_ok=True)
        return save_directory / file_name

    def _common_save_logic(
        self,
        data: np.ndarray,
        uri: Path,
        dim_order: str,
        channel_names: list[str],
        image_name: str | list[str | None] | None,
        result_str: str,
    ) -> None:
        """
        Save data as OME-TIFF with bioio based on common logic.

        Converts labels to np.int32 if np.int64 is detected, due to bioio
        not supporting np.int64 labels, even though napari and other libraries
        generate np.int64 labels.

        Parameters
        ----------
        data : np.ndarray
            The data to save.
        uri : Path
            The URI to save the data to.
        dim_order : str
            The dimension order.
        channel_names : list[str]
            The channel names saved to OME metadata
        image_name : str | list[str | None] | None
            The image name saved to OME metadata
        result_str : str
            The string used for the result widget.

        """
        # TODO: add image_name to save method
        from bioio.writers import OmeTiffWriter

        # BioImage does not allow saving labels as np.int64
        # napari generates labels differently depending on the OS
        # so we need to convert to np.int32 in case np.int64 generated
        # see: https://github.com/napari/napari/issues/5545
        # This is a failsafe
        if data.dtype == np.int64:
            data = data.astype(np.int32)

        try:
            OmeTiffWriter.save(
                data=data,
                uri=uri,
                dim_order=dim_order or None,
                channel_names=channel_names or None,
                image_name=image_name or None,
                physical_pixel_sizes=self.p_sizes,
            )
            self._results.value = f'Saved {result_str}: ' + str(
                self._save_name.value
            ) + f'\nAt {time.strftime("%H:%M:%S")}'
        # if ValueError is raised, save with default channel names
        except ValueError as e:
            OmeTiffWriter.save(
                data=data,
                uri=uri,
                dim_order=dim_order,
                image_name=image_name or None,
                physical_pixel_sizes=self.p_sizes,
            )
            self._results.value = (
                'ValueError: '
                + str(e)
                + '\nSo, saved with default channel names: \n'
                + str(self._save_name.value)
                + f'\nAt {time.strftime("%H:%M:%S")}'
            )
        return

    def _determine_save_directory(self, save_dir: str | None = None) -> str:
        if self._save_directory_prefix.value != '':
            save_dir = f'{self._save_directory_prefix.value}_{save_dir}'
        else:
            save_dir = f'{save_dir}'
        return save_dir

    def save_files_as_ome_tiff(self) -> np.ndarray:
        """Save the selected files as OME-TIFF using BioImage."""
        img_data = self.concatenate_files(self._files.value)
        save_dir = self._determine_save_directory('ConcatenatedImages')
        img_save_name = f'{self._save_name.value}.tiff'
        img_save_loc = self._get_save_loc(
            self._save_directory.value,
            save_dir,
            img_save_name,
        )

        cnames = self._channel_names.value
        channel_names = ast.literal_eval(cnames) if cnames else None

        self._common_save_logic(
            data=img_data,
            uri=img_save_loc,
            dim_order='TCZYX',
            channel_names=channel_names,
            image_name=self._save_name.value,
            result_str='Concatenated Image',
        )

        return img_data

    def batch_concatenate_files(self) -> None:
        """
        Concatenate files in the selected directory.

        Save the concatenated files as OME-TIFF, then select the next set of
        files in the directory to be concatenated. This is done by iterating
        over the remaining files in the directory based on the number of files
        selected. The files are sorted alphabetically and numerically. The
        files will be concatenated until no more files are left in the parent
        directory.
        """
        # get total number of sets of files in the directory
        parent_dir = self._files.value[0].parent
        total_num_files = len(list(parent_dir.glob(f'*{self._files.value[0].suffix}')))
        num_files = self._files.value.__len__()
        num_file_sets = total_num_files // num_files

        # check if channel names and scale are different than in the first file
        # if so, turn off the update options
        first_image = nImage(self._files.value[0])
        if first_image.channel_names != self._channel_names.value:
            self._update_channel_names.value = False
        if first_image.physical_pixel_sizes != self.p_sizes:
            self._update_scale.value = False


        # save first set of files
        self.save_files_as_ome_tiff()
        # iterate through the remaining sets of files in the directory
        for _ in range(num_file_sets):
            self.select_next_images()
            self.save_files_as_ome_tiff()

        self._results.value = (
            'Batch concatenated files in directory.'
            f'\nAt {time.strftime("%H:%M:%S")}'
        )

    def save_scenes_ome_tiff(self) -> None:
        """
        Save selected scenes as OME-TIFF.

        This method is intended to save scenes from a single file. The scenes
        are extracted based on the scenes_to_extract widget value, which is a
        list of scene indices. If the widget is left blank, then all scenes
        will be extracted.

        """
        img = nImage(self._files.value[0])

        scenes = self._scenes_to_extract.value
        scenes_list = ast.literal_eval(scenes) if scenes else img.scenes
        save_dir = self._determine_save_directory('ExtractedScenes')
        save_directory = self._save_directory.value / save_dir
        save_directory.mkdir(parents=False, exist_ok=True)

        for scene in scenes_list:
            # TODO: fix this to not have an issue if there are identical scenes
            # presented as strings, though the asssumption is most times the
            # user will input a list of integers.
            img.set_scene(scene)

            base_save_name = self._save_name.value.split('.')[0]
            image_id = helpers.create_id_string(img, base_save_name)

            img_save_name = f'{image_id}.tiff'
            img_save_loc = save_directory / img_save_name

            # get channel names from widget if truthy
            cnames = self._channel_names.value
            channel_names = ast.literal_eval(cnames) if cnames else None

            self._common_save_logic(
                data=img.data,
                uri=img_save_loc,
                dim_order='TCZYX',
                channel_names=channel_names,
                image_name=image_id,
                result_str=f'Scene: {img.current_scene}',
            )

        self._results.value = (
            f'Saved extracted scenes: {scenes_list}'
            f'\nAt {time.strftime("%H:%M:%S")}'
        )
        return

    def canvas_export_figure(self) -> None:
        """Export the current canvas figure to the save directory."""
        if self._viewer.dims.ndisplay != 2:
            self._results.value = (
                'Exporting Figure only works in 2D mode.'
                '\nUse Screenshot for 3D figures.'
                f'\nAt {time.strftime("%H:%M:%S")}'
            )
            return

        save_name = f'{self._save_name.value}_figure.png'
        save_path = self._get_save_loc(
            self._save_directory.value,
            'Figures',
            save_name,
        )

        scale = self._settings.CANVAS_SCALE

        self._viewer.export_figure(
            path=str(save_path),
            scale_factor=scale,
        )

        self._results.value = (
            f'Exported canvas figure to Figures directory.'
            f'\nSaved as {save_name}'
            f'\nWith scale factor of {scale}'
            f'\nAt {time.strftime("%H:%M:%S")}'
        )
        return

    def canvas_screenshot(self) -> None:
        """Export the current canvas screenshot to the save directory."""
        save_name = f'{self._save_name.value}_canvas.png'
        save_path = self._get_save_loc(
            self._save_directory.value,
            'Figures',
            save_name
        )

        scale = self._settings.CANVAS_SCALE
        if self._settings.OVERRIDE_CANVAS_SIZE:
            canvas_size = self._settings.CANVAS_SIZE
        else:
            canvas_size = self._viewer.window._qt_viewer.canvas.size

        self._viewer.screenshot(
            canvas_only=True,
            size=canvas_size,
            scale=scale,
            path=str(save_path),
        )

        self._results.value = (
            f'Exported screenshot of canvas to Figures directory.'
            f'\nSaved as {save_name}'
            f'\nWith canvas dimensions of {canvas_size}'
            f'\nWith scale factor of {scale}'
            f'\nAt {time.strftime("%H:%M:%S")}'
        )
        return

    def save_layers_as_ome_tiff(self) -> np.ndarray:
        """
        Save the selected layers as OME-TIFF.

        Determines types of layers and saves to corresponding directories.
        """
        layer_data = self.concatenate_layers(
            list(self._viewer.layers.selection)
        )
        # get the types of layers, to know where to save the image
        layer_types = [
            type(layer).__name__ for layer in self._viewer.layers.selection
        ]

        # if there are multiple layer types, save to Layers directory
        layer_save_type = 'Layers' if len(set(layer_types)) > 1 else layer_types[0]
        layer_save_dir = self._determine_save_directory(layer_save_type)
        layer_save_name = f'{self._save_name.value}.tiff'
        layer_save_loc = self._get_save_loc(
            self._save_directory.value, layer_save_dir, layer_save_name
        )

        # only get channel names if layer_save_type is not shapes or labels layer
        if layer_save_type not in ['Shapes', 'Labels']:
            cnames = self._channel_names.value
            channel_names = ast.literal_eval(cnames) if cnames else None
        else:
            channel_names = [layer_save_type]

        if layer_save_type == 'Shapes':
            layer_data = layer_data.astype(np.int16)

        elif layer_save_type == 'Labels':
            if layer_data.max() > 65535:
                layer_data = layer_data.astype(np.int32)
            else:
                layer_data = layer_data.astype(np.int16)

        if self._squeezed_dims_order:
            dim_order = 'C' + self._squeezed_dims_order
        else:
            # get the dim order from 'C' + whatever other dims are left but from
            # right to left of layer_data.shape
            num_dims = len(layer_data.shape)
            dim_order = 'C' + ''.join(
                [str(d) for d in 'TZYX'[-(num_dims-1):]]
            )

        self._common_save_logic(
            data=layer_data,
            uri=layer_save_loc,
            dim_order=dim_order,
            channel_names=channel_names,
            image_name=self._save_name.value,
            result_str=layer_save_type,
        )

        return layer_data
