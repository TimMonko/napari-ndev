from __future__ import annotations

import ast
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
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

from napari_ndev import helpers

if TYPE_CHECKING:
    from bioio import BioImage

    import napari
    from napari.layers import Image as ImageLayer


class UtilitiesContainer(Container):
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
    _img_data: numpy.ndarray or None
        The concatenated image data.
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
        super().__init__()

        self._viewer = viewer if viewer is not None else None
        self._img_data = None
        self._image_save_dims = None
        self._label_save_dims = None
        self._p_sizes = None
        self._squeezed_dims = None

        self._init_widgets()
        self._init_open_image_container()
        self._init_info_container()
        self._init_concatenate_container()
        self._init_save_container()
        self._init_scene_container()
        self._init_scale_container()
        self._init_layout()
        self._connect_events()

    def _init_widgets(self):
        """Initialize non-Container widgets."""
        self._file_metadata_update = PushButton(label='File')
        self._layer_metadata_update = PushButton(label='Selected Layer')
        self._metadata_container = Container(
            layout='horizontal', label='Update Metadata from'
        )
        self._metadata_container.append(self._layer_metadata_update)
        self._metadata_container.append(self._file_metadata_update)

        self._files = FileEdit(
            label='File(s)',
            mode='rm',
            tooltip='Select file(s) to load.',
        )

        self._save_directory = FileEdit(
            label='Save Directory',
            mode='d',
            tooltip='Directory where images will be saved.',
        )
        self._save_name = LineEdit(
            label='File Save Name',
            tooltip='Name of saved file. Helpful to include a'
            '.ome/.tif/.tiff extension.',
        )

        self._channel_names = LineEdit(
            label='Channel Name(s)',
            tooltip='Enter channel names as a list. If left blank or the '
            'channel names are not the proper length, then default channel '
            'names will be used.',
        )

        self._results = TextEdit(label='Info')

    def _init_open_image_container(self):
        """Initialize the open image container."""
        self._open_image_container = Container(layout='horizontal')

        self._open_image_update_metadata = CheckBox(
            value=True,
            label='Update Metadata',
            tooltip='Update metadata during initial file selection.',
        )
        self._open_image_button = PushButton(label='Open File(s)')
        self._select_next_image_button = PushButton(
            label='Select Next',
            tooltip='Select the next file(s) in the directory. '
            'The files are sorted alphabetically and numerically,'
            'which may not be consistent '
            'with your file viewer. But, opening related consecutive files '
            'should work as expected.',
        )

        self._open_image_container.append(self._open_image_update_metadata)
        self._open_image_container.append(self._open_image_button)
        self._open_image_container.append(self._select_next_image_button)

    def _init_info_container(self):
        """Initialize the info container containing dims and scenes."""
        self._info_container = Container(layout='horizontal')
        self._dim_order = Label(
            label='Dimension Order: ',
            tooltip='Sanity check for available dimensions.',
        )
        self._scenes = Label(
            label='Number of Scenes: ',
        )

        self._info_container.append(self._dim_order)
        self._info_container.append(self._scenes)

    def _init_scale_container(self):
        """Initialize the scale container."""
        self._scale_container = Container(
            layout='vertical',
            label='Scale, ZYX',
            tooltip='Pixel size, usually in μm',
        )

        self._scale_tuple = TupleEdit(
            value=(0.0000, 1.0000, 1.0000),
            options={'step': 0.0001},
        )
        self._scale_layers = PushButton(
            label='Scale Layer(s)',
            tooltip='Scale the selected layer(s) based on the given scale.',
        )
        self._scale_container.append(self._scale_tuple)
        self._scale_container.append(self._scale_layers)

    def _init_scene_container(self):
        """Initialize the scene container, allowing scene saving."""
        self._scene_container = Container(
            layout='horizontal',
            label='Extract Scenes',
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

    def _init_concatenate_container(self):
        """Initialize the container to concatenate image and layers."""
        self._concatenate_image_files = CheckBox(
            value=True,
            label='Concatenate Files',
            tooltip='Concatenate files in the selected directory. Removes '
            'blank channels.',
        )
        self._concatenate_image_layers = CheckBox(
            label='Concatenate Image Layers',
            tooltip='Concatenate image layers in the viewer. Removes empty.',
        )
        self._concatenate_container = Container(
            layout='horizontal',
            label='Image Save Options',
        )
        self._concatenate_container.append(self._concatenate_image_files)
        self._concatenate_container.append(self._concatenate_image_layers)

    def _init_save_container(self):
        """Initialize the container to save images, labels, and shapes."""
        self._save_container = Container(
            layout='horizontal',
            label='Save Selected Layers',
        )

        self._save_image_button = PushButton(
            label='Images',
            tooltip='Save the concatenated image data as OME-TIFF.',
        )
        self._save_labels_button = PushButton(
            label='Labels', tooltip='Save the labels data as OME-TIFF.'
        )
        self._save_shapes_button = PushButton(
            label='Shapes as Labels',
            tooltip='Save the shapes data as labels (OME-TIFF) according to '
            'selected image layer dimensions.',
        )

        self._save_container.append(self._save_image_button)
        self._save_container.append(self._save_labels_button)
        self._save_container.append(self._save_shapes_button)

    def _init_layout(self):
        """Initialize the layout of the widget."""
        self.extend(
            [
                self._save_directory,
                self._files,
                self._open_image_container,
                self._save_name,
                self._metadata_container,
                self._info_container,
                self._channel_names,
                self._scale_container,
                self._scene_container,
                self._concatenate_container,
                self._save_container,
                self._results,
            ]
        )

    def _connect_events(self):
        """Connect the events of the widgets to respective methods."""
        self._files.changed.connect(self.update_metadata_from_file)
        self._open_image_button.clicked.connect(self.open_images)
        self._select_next_image_button.clicked.connect(self.select_next_images)
        self._layer_metadata_update.clicked.connect(
            self.update_metadata_from_layer
        )
        self._file_metadata_update.clicked.connect(
            self.update_metadata_from_file
        )
        self._scale_layers.clicked.connect(self.rescale_by)
        self._extract_scenes.clicked.connect(self.save_scenes_ome_tiff)
        self._save_image_button.clicked.connect(self.save_ome_tiff)
        self._save_labels_button.clicked.connect(self.save_labels)
        self._save_shapes_button.clicked.connect(self.save_shapes_as_labels)
        self._results._on_value_change()

    def _update_metadata(self, img: BioImage):
        """
        Update the metadata based on the given image.

        Parameters
        ----------
        img : BioImage
            The image from which to update the metadata.

        """
        self._dim_order.value = img.dims.order

        self._squeezed_dims = helpers.get_squeezed_dim_order(img)
        self._channel_names.value = helpers.get_channel_names(img)

        self._scale_tuple.value = (
            img.physical_pixel_sizes.Z or 1,
            img.physical_pixel_sizes.Y or 1,
            img.physical_pixel_sizes.X or 1,
        )

    def _read_image_file(self, file: str | Path) -> BioImage:
        """
        Read the image file with BioImage.

        Parameters
        ----------
        file : str or Path
            The file path.

        Returns
        -------
        BioImage
            The image object.

        """
        from bioio import BioImage
        from bioio_base.exceptions import UnsupportedFileFormatError

        try:
            img = BioImage(file)
        except UnsupportedFileFormatError:
            return None
        return img

    def _bioimage_metadata(self):
        """Update the metadata from the selected file."""
        from napari_ndev.helpers import get_Image

        img = get_Image(self._files.value[0])

        self._img = img
        self._update_metadata(img)
        self._scenes.value = len(img.scenes)

    def update_metadata_from_file(self):
        """Update self._save_name.value and metadata if selected."""
        self._save_name.value = str(self._files.value[0].stem + '.tiff')

        if self._open_image_update_metadata.value:
            self._bioimage_metadata()

    def update_metadata_from_layer(self):
        """
        Update metadata from the selected layer.

        Current code expects images to be opened with napari-bioio.
        """
        selected_layer = self._viewer.layers.selection.active
        try:
            self._update_metadata(selected_layer.metadata['bioimage'])
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
                ' because layer not opened with napari-bioio'
                f'\nAt {time.strftime("%H:%M:%S")}'
            )

    def open_images(self):
        """
        Open the selected images in the napari viewer.

        If napari-bioio is not installed, then the images will likely
        be opened by the base napari reader, or a different compatabible
        reader.

        """
        self._viewer.open(self._files.value, plugin='napari-ndev')

    def select_next_images(self):
        """Open the next set of images in the directyory."""
        from napari_ndev.helpers import get_Image

        # sort the files naturally (case-insensitive and numbers in order)
        num_files = self._files.value.__len__()

        # get the parent directory of the first file
        first_file = self._files.value[0]
        parent_dir = first_file.parent

        # get the list of files in the parent directory
        files = list(parent_dir.glob(f'*{first_file.suffix}'))

        # sort the files naturally (case-insensitive and numbers in order)
        # like would be scene in windows file explorer default sorting
        files.sort(
            key=lambda f: [
                int(text) if text.isdigit() else text.lower()
                for text in re.split('([0-9]+)', f.name)
            ]
        )

        # get the index of the first file in the list and then the next files
        idx = files.index(first_file)
        next_files = files[idx + num_files : idx + num_files + num_files]
        # set the nwe save names, and update the file value
        img = get_Image(next_files[0])

        image_id = helpers.create_id_string(img, next_files[0].stem)
        self._save_name.value = str(image_id + '.tiff')
        self._files.value = next_files

        if self._open_image_update_metadata.value:
            self._bioimage_metadata()

    def rescale_by(self):
        """Rescale the selected layers based on the given scale."""
        layers = self._viewer.layers.selection
        scale_tup = self._scale_tuple.value

        for layer in layers:
            scale_len = len(layer.scale)
            # get the scale_tup from the back of the tuple first, in case dims
            # are missing in the new layer
            layer.scale = scale_tup[-scale_len:]

    def concatenate_images(
        self,
        concatenate_files: bool,
        files: list[str | Path],
        concatenate_layers: bool,
        layers: list[ImageLayer],
    ):
        """
        Concatenate the image data based on the selected options.

        Intended also to remove "blank" channels and layers. This is present
        in some microscope formats where it will image in RGB, and then
        leave empty channels not represented by the color channels.

        Parameters
        ----------
        concatenate_files : bool
            Concatenate files in the selected directory. Removes blank channels.
        files : list[str | Path]
            The list of files to concatenate.
        concatenate_layers : bool
            Concatenate image layers in the viewer. Removes empty.
        layers : list[ImageLayer]
            The list of layers to concatenate.

        Returns
        -------
        numpy.ndarray
            The concatenated image data.

        """
        from napari_ndev.helpers import get_Image

        array_list = []
        if concatenate_files:
            for file in files:
                img = get_Image(file)

                if 'S' in img.dims.order:
                    img_data = img.get_image_data('TSZYX')
                else:
                    img_data = img.data

                # iterate over all channels and only keep if not blank
                for idx in range(img_data.shape[1]):
                    array = img_data[:, [idx], :, :, :]
                    if array.max() > 0:
                        array_list.append(array)

        # <- fix if RGB image is the layer data
        if concatenate_layers:
            for layer in layers:
                layer_data = layer.data
                # convert to 5D array for compatability with image dims
                while len(layer_data.shape) < 5:
                    layer_data = np.expand_dims(layer_data, axis=0)
                array_list.append(layer_data)

        return np.concatenate(array_list, axis=1)

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
            The parent directory. eg. 'Images', 'Labels', 'ShapesAsLabels'
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
        layer: str,
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
        layer : str
            The layer name.

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
            self._results.value = f'Saved {layer}: ' + str(
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

    def save_scenes_ome_tiff(self) -> None:
        """
        Save selected scenes as OME-TIFF.

        Not currently interacting with the viewer labels.
        This method is intended to save scenes from a single file. The scenes
        are extracted based on the scenes_to_extract widget value, which is a
        list of scene indices. If the widget is left blank, then all scenes
        will be extracted.

        """
        from napari_ndev.helpers import get_Image

        img = get_Image(self._files.value[0])

        scenes = self._scenes_to_extract.value
        scenes_list = ast.literal_eval(scenes) if scenes else None
        save_directory = self._save_directory.value / 'Images'
        save_directory.mkdir(parents=False, exist_ok=True)
        for scene in scenes_list:
            # TODO: fix this to not have an issue if there are identical scenes
            # presented as strings, though the asssumption is most times the
            # user will input a list of integers.
            img.set_scene(scene)

            base_save_name = self._save_name.value.split('.')[0]
            image_id = helpers.create_id_string(img, base_save_name)

            img_save_name = f'{image_id}.ome.tiff'
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
                layer=f'Scene: {img.current_scene}',
            )
        return

    def save_ome_tiff(self) -> np.ndarray:
        """
        Save the concatenated image data as OME-TIFF.

        The concatenated image data is concatenated and then saved as OME-TIFF
        based on the selected options and the given save location.

        Returns
        -------
        numpy.ndarray
            The concatenated image data.

        """
        self._img_data = self.concatenate_images(
            self._concatenate_image_files.value,
            self._files.value,
            self._concatenate_image_layers.value,
            list(self._viewer.layers.selection),
        )
        img_save_loc = self._get_save_loc(
            self._save_directory.value, 'Images', self._save_name.value
        )
        # get channel names from widget if truthy
        cnames = self._channel_names.value
        channel_names = ast.literal_eval(cnames) if cnames else None

        self._common_save_logic(
            data=self._img_data,
            uri=img_save_loc,
            dim_order='TCZYX',
            channel_names=channel_names,
            image_name=self._save_name.value,
            layer='Image',
        )
        return self._img_data

    def save_labels(self) -> np.ndarray:
        """
        Save the selected labels layer as OME-TIFF.

        Returns
        -------
        numpy.ndarray
            The labels data.

        """
        label_data = self._viewer.layers.selection.active.data

        if label_data.max() > 65535:
            label_data = label_data.astype(np.int32)
        else:
            label_data = label_data.astype(np.int16)

        label_save_loc = self._get_save_loc(
            self._save_directory.value, 'Labels', self._save_name.value
        )

        self._common_save_logic(
            data=label_data,
            uri=label_save_loc,
            dim_order=self._squeezed_dims,
            channel_names=['Labels'],
            image_name=self._save_name.value,
            layer='Labels',
        )
        return label_data

    def save_shapes_as_labels(self) -> np.ndarray:
        """
        "Save the selected shapes layer as labels.

        Returns
        -------
        numpy.ndarray
            The shapes data as labels.

        """
        from napari.layers import Image as ImageLayer

        # inherit shape from selected image layer or else a default
        image_layers = [
            x for x in self._viewer.layers if isinstance(x, ImageLayer)
        ]
        label_dim = image_layers[0].data.shape

        # drop last axis if represents RGB image
        label_dim = label_dim[:-1] if label_dim[-1] == 3 else label_dim

        shapes = self._viewer.layers.selection.active
        shapes_as_labels = shapes.to_labels(labels_shape=label_dim)
        shapes_as_labels = shapes_as_labels.astype(np.int16)

        shapes_save_loc = self._get_save_loc(
            self._save_directory.value, 'ShapesAsLabels', self._save_name.value
        )

        self._common_save_logic(
            data=shapes_as_labels,
            uri=shapes_save_loc,
            dim_order=self._squeezed_dims,
            channel_names=['Shapes'],
            image_name=self._save_name.value,
            layer='Shapes',
        )

        return shapes_as_labels
