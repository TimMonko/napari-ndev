from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    Label,
    ProgressBar,
    PushButton,
    Select,
)
from qtpy.QtWidgets import QTabWidget

from napari_ndev import helpers

if TYPE_CHECKING:
    import napari


class WorkflowContainer(Container):
    """
    Container class for managing the workflow functionality in napari-ndev.

    Parameters
    ----------
    viewer : napari.viewer.Viewer
        The napari viewer instance.

    Attributes
    ----------
    viewer : napari.viewer.Viewer
        The napari viewer instance.
    roots : list
        List of ComboBox widgets representing the workflow roots.
    _channel_names : list
        List of channel names extracted from the image data.
    _img_dims : str
        The dimensions of the image data.

    Widgets:
    --------
    image_directory : FileEdit
        Widget for selecting the image directory.
    result_directory : FileEdit
        Widget for selecting the result directory.
    workflow_file : FileEdit
        Widget for selecting the workflow file.
    _keep_original_images : CheckBox
        Checkbox widget for specifying whether to keep original images.
    batch_button : PushButton
        Button widget for triggering the batch workflow.
    _progress_bar : ProgressBar
        Progress bar widget for displaying the progress of the workflow.
    _workflow_roots : Label
        Label widget for displaying the workflow roots.

    Events:
    -------
    image_directory.changed : Signal
        Signal emitted when the image directory is changed.
    workflow_file.changed : Signal
        Signal emitted when the workflow file is changed.
    batch_button.clicked : Signal
        Signal emitted when the batch button is clicked.

    """

    def __init__(self, viewer: napari.viewer.Viewer = None):
        """
        Initialize the WorkflowContainer widget.

        Parameters
        ----------
        viewer : napari.viewer.Viewer, optional
            The napari viewer instance.

        """
        super().__init__()
        self.viewer = viewer if viewer is not None else None
        self.roots = []
        self._channel_names = []
        self._img_dims = ''

        self._init_widgets()
        self._roots_container()
        self._tasks_container()
        self._init_layout()
        self._connect_events()

    def _init_widgets(self):
        """Initialize non-Container widgets."""
        self.image_directory = FileEdit(label='Image Directory', mode='d')
        self.result_directory = FileEdit(label='Result Directory', mode='d')

        self.workflow_file = FileEdit(
            label='Workflow File',
            filter='*.yaml',
            tooltip='Select a workflow file to load',
        )
        self._keep_original_images = CheckBox(
            label='Keep Original Images',
            value=False,
            tooltip='If checked, the original images will be '
            'concatenated with the results',
        )
        self.batch_button = PushButton(label='Batch Workflow')

        self._progress_bar = ProgressBar(label='Progress:')
        self._workflow_roots = Label(label='Workflow Roots:')

    def _roots_container(self):
        """Initialize the roots container."""
        self._roots_container = Container(layout='vertical')

    def _tasks_container(self):
        """Initialize the tasks container."""
        self._tasks_container = Container(layout='vertical')

        self._tasks_select = Select(
            choices=[],
            nullable=False,
            allow_multiple=True,
        )

        self._tasks_container.append(self._tasks_select)

    def _init_layout(self):
        """Initialize the layout of the widgets."""
        self.extend(
            [
                self.image_directory,
                self.result_directory,
                self.workflow_file,
                self._keep_original_images,
                self.batch_button,
                self._progress_bar,
                self._workflow_roots,
            ]
        )

        tabs = QTabWidget()
        tabs.addTab(self._roots_container.native, 'Roots')
        tabs.addTab(self._tasks_container.native, 'Tasks')
        self.native.layout().addWidget(tabs)

    def _connect_events(self):
        """Connect the events of the widgets to respective methods."""
        self.image_directory.changed.connect(self._get_image_info)
        self.workflow_file.changed.connect(self._get_workflow_info)
        self.batch_button.clicked.connect(self.batch_workflow)

    def _get_image_info(self):
        """Get channels and dims from first image in the directory."""
        self.image_dir, self.image_files = helpers.get_directory_and_files(
            self.image_directory.value,
        )
        img = helpers.get_Image(self.image_files[0])

        self._channel_names = helpers.get_channel_names(img)

        for widget in self._roots_container:
            widget.choices = self._channel_names

        self._squeezed_img_dims = helpers.get_squeezed_dim_order(img)
        return self._squeezed_img_dims

    def _update_roots(self):
        """Get the roots from the workflow and update the ComboBox widgets."""
        self._roots_container.clear()

        for idx, root in enumerate(self.workflow.roots()):
            root_combo = ComboBox(
                label=f'Root {idx}: {root}',
                choices=self._channel_names,
                nullable=True,
                value=None,
            )
            self._roots_container.append(root_combo)
            # self.append(root_combo)
        return

    def _update_task_choices(self, workflow):
        """Update the choices of the tasks with the workflow tasks."""
        self._tasks_select.choices = list(workflow._tasks.keys())
        self._tasks_select.value = workflow.leafs()

    def _get_workflow_info(self):
        """Load the workflow file and update the roots and leafs."""
        from napari_workflows._io_yaml_v1 import load_workflow

        self.workflow = load_workflow(self.workflow_file.value)
        self._workflow_roots.value = self.workflow.roots()
        self._update_roots()
        self._update_task_choices(self.workflow)
        return

    def batch_workflow(self):
        """Run the workflow on all images in the image directory."""
        import dask.array as da
        from bioio.writers import OmeTiffWriter
        from bioio_base import transforms

        result_dir = self.result_directory.value
        image_files = self.image_files
        workflow = self.workflow

        # get indexes of channel names, in case not all images have
        # the same channel names, the index should be in the same order
        root_list = [widget.value for widget in self._roots_container]
        root_index_list = [self._channel_names.index(r) for r in root_list]

        # Setting up Logging File
        log_loc = result_dir / 'workflow.log.txt'
        logger, handler = helpers.setup_logger(log_loc)
        logger.info(
            """
            Image Directory: %s
            Result Directory: %s
            Workflow File: %s
            Roots: %s
            Tasks: %s
            """,
            self.image_directory.value,
            result_dir,
            self.workflow_file.value,
            root_list,
            self._tasks_select.value,
        )

        self._progress_bar.label = f'Workflow on {len(image_files)} images'
        self._progress_bar.value = 0
        self._progress_bar.max = len(image_files)

        for idx_file, image_file in enumerate(image_files):
            logger.info('Processing %d: %s', idx_file + 1, image_file.name)
            img = helpers.get_Image(image_file)

            root_stack = []
            # get image corresponding to each root, and set it to the workflow
            for idx, root_index in enumerate(root_index_list):
                if 'S' in img.dims.order:
                    root_img = img.get_image_data('TSZYX', S=root_index)
                else:
                    root_img = img.get_image_data('TCZYX', C=root_index)
                # stack the TCZYX images for later stacking with results
                root_stack.append(root_img)
                # squeeze the root image for workflow
                root_squeeze = np.squeeze(root_img)
                # set the root image to the index of the root in the workflow
                workflow.set(
                    name=workflow.roots()[idx], func_or_data=root_squeeze
                )

            task_names = self._tasks_select.value
            result = workflow.get(name=task_names)

            result_stack = np.asarray(
                result
            )  # cle.pull stacks the results on the 0th axis as "C"
            # transform result_stack to TCZYX
            result_stack = transforms.reshape_data(
                data=result_stack,
                given_dims='C' + self._squeezed_img_dims,
                return_dims='TCZYX',
            )

            if result_stack.dtype == np.int64:
                result_stack = result_stack.astype(np.int32)

            # <- should I add a check for the result_stack to be a dask array?
            # <- should this be done using dask or numpy?
            if self._keep_original_images.value:
                dask_images = da.concatenate(root_stack, axis=1)  # along "C"
                result_stack = da.concatenate(
                    [dask_images, result_stack], axis=1
                )
                result_names = root_list + task_names
            else:
                result_names = task_names

            OmeTiffWriter.save(
                data=result_stack,
                uri=result_dir / (image_file.stem + '.tiff'),
                dim_order='TCZYX',
                channel_names=result_names,
                image_name=image_file.stem,
                physical_pixel_sizes=img.physical_pixel_sizes,
            )

            self._progress_bar.value = idx_file + 1

        logger.removeHandler(handler)
        return
