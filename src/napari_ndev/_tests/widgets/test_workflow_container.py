import pathlib

from napari_ndev import helpers
from napari_ndev.widgets._workflow_container import WorkflowContainer

# from napari_workflows._io_yaml_v1 import load_workflow

class MockWorkflow:
    def roots(self):
        return ['root1', 'root2']

    def leafs(self):
        return ['leaf1', 'leaf2']

    def set(self, name, func_or_data):
        pass

    def get(self, name):
        pass

def test_workflow_container_init_no_viewer():
    container = WorkflowContainer()

    assert container.viewer is None
    assert container.roots == []
    assert container._channel_names == []
    assert container._img_dims == ''

def test_workflow_container_init_with_viewer(make_napari_viewer):
    viewer = make_napari_viewer()
    container = WorkflowContainer(viewer)

    assert container.viewer == viewer
    assert container.roots == []
    assert container._channel_names == []
    assert container._img_dims == ''


def test_workflow_container_update_roots():
    container = WorkflowContainer()

    container.workflow = MockWorkflow()
    container._channel_names = ['red', 'green', 'blue']

    container._update_roots()
    # check that there are two widgets created
    assert len(container._roots_container) == 2

    for idx, root in enumerate(container._roots_container):
        assert root.label == f'Root {idx}: {container.workflow.roots()[idx]}'
        assert root.choices == (None, 'red', 'green', 'blue')
        assert root._nullable is True
        assert root.value is None


def test_workflow_container_get_workflow_info():
    container = WorkflowContainer()
    wf_path = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/workflows/'
        'cpu_workflow-2roots-2leafs.yaml'
    )
    container.workflow_file.value = wf_path

    assert container._workflow_roots.value == str(container.workflow.roots())
    assert len(container._roots_container) == len(container.workflow.roots())
    assert container._tasks_select.value == list(container.workflow.leafs())
    assert list(container._tasks_select.choices) == list(container.workflow._tasks.keys())



def test_batch_workflow_leaf_tasks(tmp_path):
    container = WorkflowContainer()
    wf_path = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/workflows/'
        'cpu_workflow-2roots-2leafs.yaml'
    )
    container.workflow_file.value = wf_path

    container.image_directory.value = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/Images'
    )

    output_folder = tmp_path / 'Output'
    output_folder.mkdir()
    container.result_directory.value = output_folder

    container._roots_container[0].value = 'membrane'
    container._roots_container[1].value = 'nuclei'

    container.batch_workflow()

    assert output_folder.exists()
    assert (output_folder / 'cells3d2ch.tiff').exists()

    img = helpers.get_Image(output_folder / 'cells3d2ch.tiff')
    assert len(img.channel_names) == 2

def test_batch_workflow_keep_original_images(tmp_path):
    container = WorkflowContainer()
    wf_path = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/workflows/'
        'cpu_workflow-2roots-2leafs.yaml'
    )
    container.workflow_file.value = wf_path

    container.image_directory.value = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/Images'
    )

    output_folder = tmp_path / 'Output'
    output_folder.mkdir()
    container.result_directory.value = output_folder

    container._roots_container[0].value = 'membrane'
    container._roots_container[1].value = 'nuclei'

    container._keep_original_images.value = True
    container.batch_button.clicked()

    assert output_folder.exists()
    assert (output_folder / 'cells3d2ch.tiff').exists()

    img = helpers.get_Image(output_folder / 'cells3d2ch.tiff')
    assert len(img.channel_names) == 4

def test_batch_workflow_all_tasks(tmp_path):
    container = WorkflowContainer()
    wf_path = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/workflows/'
        'cpu_workflow-2roots-2leafs.yaml'
    )
    container.workflow_file.value = wf_path

    container.image_directory.value = pathlib.Path(
        'src/napari_ndev/_tests/resources/Workflow/Images'
    )

    output_folder = tmp_path / 'Output'
    output_folder.mkdir()
    container.result_directory.value = output_folder

    container._roots_container[0].value = 'membrane'
    container._roots_container[1].value = 'nuclei'

    container._tasks_select.value = list(container.workflow._tasks.keys())

    container.batch_workflow()

    assert output_folder.exists()
    assert (output_folder / 'cells3d2ch.tiff').exists()

    img = helpers.get_Image(output_folder / 'cells3d2ch.tiff')
    assert len(img.channel_names) == 6
