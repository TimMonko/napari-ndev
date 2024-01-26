# napari-ndev

[![License BSD-3](https://img.shields.io/pypi/l/napari-ndev.svg?color=green)](https://github.com/TimMonko/napari-ndev/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-ndev.svg?color=green)](https://pypi.org/project/napari-ndev)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-ndev.svg?color=green)](https://python.org)
[![tests](https://github.com/TimMonko/napari-ndev/workflows/tests/badge.svg)](https://github.com/TimMonko/napari-ndev/actions)
[![codecov](https://codecov.io/gh/TimMonko/napari-ndev/branch/main/graph/badge.svg)](https://codecov.io/gh/TimMonko/napari-ndev)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-ndev)](https://napari-hub.org/plugins/napari-ndev)

A collection of widgets intended to serve any person seeking to process microscopy images from start to finish. The wide breadth of this plugin's scope is only made possible by the amazing libraries and plugins from the napari community, especially Robert Haase. Currently, the plugin supports the following goals:

1. **Image-utilities:** Allows opening image files (using aics-imageio) and displaying in napari. Also reads metadata and allows customization prior to saving images and labels layers. Allows concatenation of image files and image layers for saving new images. Speeds up annotation by saving corresponding images and labels in designated folders. Also allows saving of shapes layers as labels in case shapes are being used as a region of interest.
2. **Batch-workflow:** Batch pre-processing/processing images using [napari-workflows].
3. **Batch-APOC:** Utilizes the excellent accelerated-pixel-and-object-classification ([apoc]) in a similar fashion to [napari-apoc], but intended for batch training and prediction with a napari widget instead of scripting.
4. **Rescale-by:** Rescale any napari layer (image, label, shape) by a set amount, which can be inherited from a different image layer's metadata.

----------------------------------

![Plugin-Abstract](/Plugin-Abstract.png)


This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-ndev` via [pip]:

    pip install napari-ndev

----------------------------------

## Further Info

### 1. Image Utilities
A quick and easy way to save annotations (a napari labels layer) and corresponding images to corresponding folders. *Requires* that images are opened with [napari-aicsimageio]--which can be as simple as drag and drop opening by setting the appropriate default reader for each file type in Preferences -> Plugins--in order to utilize the metadata present for saving the image-label pairs. (See Note about AICSImageIO)

Quick uniform adjustments to a folder of images, saving the output. Currently supports selecting channels, slicing Z, cropping/downsampling in XY, and doing a max projection of the sliced/cropped image data. To be added: alternative projection types, slicing in T, and compatability with non TCZYX images (but this is not a priority since [aicsimageio] currently always extracts images as TCZYX even if a dim is only length 1.

### 2. Batch-workflow
Batch pre-processing/processing images using [napari-workflows].  Images are processed outside the napari-viewer using [aicsimageio] as both reader and writer. Prior to passing the images to napari-workflows, the user selects the correct images as the roots (inputs) and thus napari-workflows matches the processing to create the outputs. The advantage of using napari-workflows for batch processing is that it provides an incredibly flexible processing interface without writing a novel widget for small changes to processing steps like specific filters, segmentation, or measurements. Currently only intended for use with images as inputs and images as outputs from napari-workflows, though there is future potential to have other outputs possible, such as .csv measurement arrays.

### 3. Batch-training/prediction
Utilizes the excellent accelerated-pixel-and-object-classification ([apoc]) in a similar fashion to [napari-apoc], but intended for batch training and prediction with a napari widget instead of scripting. Recognizes pre established feature set, and custom feature sets (a string of filters and radii) can be genereated with a corresponding widget.

### A Note about AICSImageIO
[AICSImageIO] is a convenient, multi-format file reader which also has the complimentary [napari-aicsimageio] reader plugin. By default, napari-aicsimageio installs all reader dependencies. Because napari-aicsimageio is not technically required for this plugin to work (you could build your own metadata for the annotation-saver) and just napari-aicsimage is required, the former is not an install requirement. This is to avoid using the GPL liscence and to stick with BSD-3. However, you should install napari-aicsimageio if you want the smoothest operation of the annotation-saver.

----------------------------------

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-ndev" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

[napari-workflows]: https://github.com/haesleinhuepf/napari-workflows
[apoc]: https://github.com/haesleinhuepf/apoc
[napari-apoc]: https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification
[napari-aicsimageio]: https://github.com/AllenCellModeling/napari-aicsimageio
[AICSImageIO]: https://allencellmodeling.github.io/aicsimageio/
