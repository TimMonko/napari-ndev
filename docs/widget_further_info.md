# Further Info

## 1. Image Utilities

A quick and easy way to save annotations (a napari labels layer) and corresponding images to corresponding folders. Best if the images are opened with the `nDev` reader (using [bioio] under the hood) -- which can be as simple as drag and drop opening by setting the appropriate default reader for each file type in Preferences -> Plugins--in order to utilize the metadata present for saving the image-label pairs.

Quick uniform adjustments to a folder of images, saving the output. Currently supports selecting channels, slicing Z, cropping/downsampling in XY, and doing a max projection of the sliced/cropped image data. To be added: alternative projection types, slicing in T, and compatibility with non TCZYX images (but this is not a priority since [bioio] currently always extracts images as TCZYX even if a dim is only length 1.

## 2. Workflow Widget

Batch pre-processing/processing images using [napari-workflows].  Images are processed outside the napari-viewer using [bioio] as both reader and writer. Prior to passing the images to napari-workflows, the user selects the correct images as the roots (inputs) and thus napari-workflows matches the processing to create the outputs. The advantage of using napari-workflows for batch processing is that it provides an incredibly flexible processing interface without writing a novel widget for small changes to processing steps like specific filters, segmentation, or measurements. Currently only intended for use with images as inputs and images as outputs from napari-workflows, though there is future potential to have other outputs possible, such as .csv measurement arrays.

## 3. APOC Widget

Utilizes the excellent accelerated-pixel-and-object-classification ([apoc]) in a similar fashion to [napari-apoc], but intended for batch training and prediction with a napari widget instead of scripting. Recognizes pre established feature set, and custom feature sets (a string of filters and radii) can be generated with a corresponding widget. Also contains a Custom Feature Set widget which allows application of all the features to a layer in the viewer, for improved visualization.

## 4. Measure Widget

Batch measurements using [scikit-image]'s [regionprops]. This can measure features of a label such as area, eccentricity, and more but also can measure various intensity metrics. Attempts to support post-processing of measurements, grouping, and more to make downstream analyses easier for users. Will be updated in the future to include [nyxus].

[napari-workflows]: https://github.com/haesleinhuepf/napari-workflows
[apoc]: https://github.com/haesleinhuepf/apoc
[napari-apoc]: https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification
[bioio]: https://github.com/bioio-devs/bioio
[scikit-image]: https://scikit-image.org/
[regionprops]: https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
[nyxus]: https://github.com/PolusAI/nyxus
