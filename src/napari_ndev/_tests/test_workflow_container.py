import pathlib

from napari_ndev._workflow_container import WorkflowContainer

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
        'test_2roots_1leaf.yaml'
    )
    container.workflow_file.value = wf_path

    assert container._workflow_roots.value == str(container.workflow.roots())
    assert len(container._roots_container) == len(container.workflow.roots())
    assert container._tasks_select.value == list(container.workflow.leafs())
    assert list(container._tasks_select.choices) == list(container.workflow._tasks.keys())
