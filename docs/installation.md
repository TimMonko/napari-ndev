# Installation

**napari-ndev** is a pure Python package, and can be installed with `pip`:

```bash
pip install napari-ndev
```

You may also like to install `napari-aicsimageio` to properly handle metadata with the `Image Utilities` widget.

```bash
pip install napari-aicsimageio
```

In addition, you may need to install specific [`bioio` readers](https://github.com/bioio-devs/bioio) to support your specific image, such as `bioio-czi` or `bioio-lif`.
