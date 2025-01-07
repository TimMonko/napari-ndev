from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from magicclass.widgets import TabbedContainer
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    LineEdit,
    ProgressBar,
    PushButton,
    Select,
)

from napari_ndev import helpers, nImage

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
        self.image_files = []
        self.workflow = None

        self._init_widgets()
        self._init_batch_container()
        self._tasks_container()
        self._init_layout()
        self._connect_events()

    def _init_widgets(self):
        """Initialize non-Container widgets."""
        self.workflow_file = FileEdit(
            label='Workflow File',
            filter='*.yaml',
            tooltip='Select a workflow file to load',
        )
        self._workflow_roots = LineEdit(label='Workflow Roots:')
        self._progress_bar = ProgressBar(label='Progress:')

    def _init_batch_container(self):
        """Initialize the batch container tab widgets."""
        # first create the informational widgets
        self.image_directory = FileEdit(label='Image Directory', mode='d')
        self.result_directory = FileEdit(label='Result Directory', mode='d')
        self._keep_original_images = CheckBox(
            label='Keep Original Images',
            value=False,
            tooltip='If checked, the original images will be '
            'concatenated with the results',
        )
        self.batch_button = PushButton(label='Batch Workflow')
        self._batch_info_container = Container(
            layout='vertical',
            widgets=[
                self.image_directory,
                self.result_directory,
                self._keep_original_images,
                self.batch_button,
            ],
        )

        # create the container where roots will later be added
        self._batch_roots_container = Container(layout='vertical', label=None)
        self._batch_roots_container.native.layout().addStretch() # this resets the additions to the top of the container (the name is confusing)

        # establish the layout of the batch container
        self._batch_container = Container(
            layout='vertical',
            widgets=[
                self._batch_info_container,
                self._batch_roots_container,
            ],
            label='Batch',
            labels=None,
        )


    def _tasks_container(self):
        """Initialize the tasks container."""
        self._tasks_select = Select(
            choices=[],
            nullable=False,
            allow_multiple=True,
        )
        self._tasks_container = Container(
            layout='vertical',
            widgets=[self._tasks_select],
            label='Tasks',
        )

    def _init_layout(self):
        """Initialize the layout of the widgets."""
        self.extend(
            [
                self.workflow_file,
                self._workflow_roots,
                self._progress_bar,
            ]
        )
        self._tabs = TabbedContainer(
            widgets=[
                self._batch_container,
                self._tasks_container,
            ],
            label=None,
            labels=None,
        )
        self.native.layout().addWidget(self._tabs.native) # add the tabbed container to the layout, native needed to keep viewer interaction
        self.native.layout().addStretch() # resets the layout to the top of the container

    def _connect_events(self):
        """Connect the events of the widgets to respective methods."""
        self.image_directory.changed.connect(self._get_image_info)
        self.workflow_file.changed.connect(self._get_workflow_info)
        self.batch_button.clicked.connect(self.batch_workflow_threaded)

    def _get_image_info(self):
        """Get channels and dims from first image in the directory."""
        self.image_dir, self.image_files = helpers.get_directory_and_files(
            self.image_directory.value,
        )
        img = nImage(self.image_files[0])

        self._channel_names = helpers.get_channel_names(img)

        for widget in self._batch_roots_container:
            widget.choices = self._channel_names

        self._squeezed_img_dims = helpers.get_squeezed_dim_order(img)
        return self._squeezed_img_dims

    def _update_roots(self):
        """Get the roots from the workflow and update the ComboBox widgets."""
        self._batch_roots_container.clear()

        for idx, root in enumerate(self.workflow.roots()):
            short_root = helpers.elide_string(root, max_length=12)
            root_combo = ComboBox(
                label=f'Root {idx}: {short_root}',
                choices=self._channel_names,
                nullable=True,
                value=None,
            )
            self._batch_roots_container.append(root_combo)
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

    def _update_progress_bar(self, value):
        self._progress_bar.value = value
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
        root_list = [widget.value for widget in self._batch_roots_container]
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

        for idx_file, image_file in enumerate(image_files):
            logger.info('Processing %d: %s', idx_file + 1, image_file.name)
            img = nImage(image_file)

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

            yield idx_file + 1

        logger.removeHandler(handler)
        return

    def batch_workflow_threaded(self):
        """Run the batch workflow with threading and progress bar updates."""
        from napari.qt import create_worker

        self._progress_bar.label = f'Workflow on {len(self.image_files)} images'
        self._progress_bar.value = 0
        self._progress_bar.max = len(self.image_files)

        self._batch_worker = create_worker(self.batch_workflow)
        self._batch_worker.yielded.connect(self._update_progress_bar)
        self._batch_worker.start()
        return
