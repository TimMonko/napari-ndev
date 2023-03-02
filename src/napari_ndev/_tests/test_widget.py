import numpy as np
from aicsimageio import AICSImage

# from napari_ndev import batch_annotator  # , batch_predict, batch_training


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_batch_annotator(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    # viewer = napari.Viewer()
    test_image = np.random.random((3, 2, 4, 100, 100))
    test_aics = AICSImage(test_image)
    viewer.add_image(test_aics.data)
    test_thresh = test_image > 0.5
    viewer.add_labels(test_thresh)

    # create our widget, passing in the viewer
    # my_widget = batch_annotator()
    # my_widget()
