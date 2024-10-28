# Easy Machine Learning

The goal of this tutorial is to get a user familiar with generating annotations, workflows, and machine learning classifiers. Unlike the [Example Pipeline Tutorial](01_example_pipeline.md), this tutorial just provides raw images and hints on how to progress.

If you investigate the `primaryneurons` images you'll notice that there variable interesting morphologies that are not easy to segment by traditional intensity based segmentation. Machine Learning fills this gap (you'll see!) that Deep Learning has yet to sort out.

You might also be surprised when looking at some of the images that I would not recommend traditional intensity-based segmentation methods for NCOA4 and Ferritin (but would, and do, use it for DAPI). Instead, I would endorse using Machine Learning based segmentation because it is less sensitive to intensity (which is expected to be different between neurons and treatment group) and more sensitive to 'Features' of the images, which includes intensity, size, blobness, ridgeness, edgeness, etc.

The skills practiced in this will be used on relatively small, 2D images; however, things are intended to generally transfer to both 3D and higher dimensional datasets.

## Sparse annotation with Image Utilities

One strength of `napari-ndev` is the ability to quickly annotate images and save them, while maintaining helpful metadata to pair the images up for future processing. In `napari` annotations can be made using `labels` or `shapes`. Shapes has a current weakness in that it cannot save `images`, so `napari-ndev` converts shapes to `labels` so that they match the image format. For this tutorial, we want to use the `labels` feature to 'draw' on annotations.

1. Load in one of the primary neuron images in `ExtractedScenes` using the `Image Utilities` widget.
2. Add a Labels layer by clicking the `tag` icon (the third button above the layer list)
3. Click the `Paintbrush` button in the `layer controls`.
4. Click and drag on the image to draw annotations.
5. Draw background labels with the original

## Generation of a Feature Set with APOC Widget

## Training a Machine Learning Classifier

## Predicting with a Machine Learning Classifier
