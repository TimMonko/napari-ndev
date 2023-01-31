import numpy as np

from napari_ndev import batch_annotator


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_batch_annotator(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    test_image = np.random.random((100, 100))
    viewer.add_image(test_image)
    test_thresh = test_image > 1
    viewer.add_labels(test_thresh)

    # create our widget, passing in the viewer
    my_widget = batch_annotator()
    my_widget()
    # call our widget method
    # my_widget._on_click()

    # read captured output and check that it's as we expected
    # captured = capsys.readouterr()
    # assert captured.out == "napari has 1 layers\n"
