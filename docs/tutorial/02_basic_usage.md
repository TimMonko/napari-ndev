# Basic Usage Tutorial

The goal of this tutorial is to get a user familiar with generating annotations, workflows, and machine learning classifiers. Unlike the [Example Pipeline Tutorial](01_example_pipeline.md), this tutorial just provides raw images and hints on how to progress.

The skills practiced in this will be used on relatively small, 2D images; however, things are intended to generally transfer to both 3D and higher dimensional datasets.

## Easy Machine Learning Setup

### Sparse annotation with Image Utilities

### Generation of a Feature Set with APOC Widget

### Training a Machine Learning Classifier

### Predicting with a Machine Learning Classifier

## Build your own Workflow

For this workflow, we will be using the `neuralprogenitors` images. Our goal is to segment the PAX6 and TBR2 channels. We also specifically want to make an ROI that is 200 microns wide on each image, and bin a specific region of the brand (the specifics beyond the scope and necessity of this tutorial). Later, we will use these labels to count only the ones inside the region of interest.

### Annotating regions of interest with Image Utilities

### Using the napari-assistant to generate a workflow

??? tip "How to label"

    You may find the functions in the `Label` button to be quite useful.

??? tip "A very useful label function"

    Check out the [voronoi_otsu_labeling](https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/20_image_segmentation/11_voronoi_otsu_labeling.html) function. Read the link for more info.

??? tip "Pre-processing the images to reduce background"

    Try playing with functions in `remove noise` and `remove background` to remove some of the variability in background intensity and off-target fluorescence prior to labeling. This will make labeling more consister.

??? tip "Cleaning up the labels"

    Perhaps you have criteria for what labels you want to keep. Check out `Process Labels` button for cleaning up things like small or large labels, or labels on the edges.

??? tip "OK, I give up, just give me the answer"

    Something like the following should work well.

    1. median_sphere (pyclesperanto) with radii of 1
    2. top_hat_sphere (pyclesperanto) with radii of 10
    3. voronoi_otsu_label (pyclesperanto) with spot and outline sigmas of 1
    4. exclude_small_labels (pyclesperanto) that are smaller than 10 pixels

### Applying your workflow in batch with the Workflow Widget

## Notes on multi-dimensional data

Overall, most of the plugin should be able to handle datasets that have time, multi-channel, and 3D data. Try exploring the `Lund Timelapse (100MB)` sample data from `Pyclesperanto` in napari.
