# Installation

**napari-ndev** is a pure Python package, and can be installed with [pip]:

```bash
pip install napari-ndev
```

----------------------------------

### Optional Libraries

**napari-ndev** is most useful when interacting with some other napari plugins (e.g. napari-assistant) and can read additional filetypes (e.g. bioio-nd2). You may install these BSD-3 compatible plugins with [pip]:

```bash
pip install napari-ndev[extras]
```

**napari-ndev** can optionally use GPL-3 licensed libraries to enhance its functionality, but are not required. If you choose to install and use these optional dependencies, you must comply with the GPL-3 license terms. The main functional improvement is from `napari-bioio` to properly handle metadata with the `Image Utilities` widget. These libraries can be installed with [pip]:

```bash
pip install napari-ndev[gpl-extras]

or

pip install napari-ndev[all]
```

In addition, you may need to install specific [`bioio` readers](https://github.com/bioio-devs/bioio) to support your specific image, such as `bioio-czi` and `bioio-lif` (included in `[gpl-extras]`) or `bioio-bioformats`.

### Development Libraries

For development use the `[dev]` optional libraries. You may also like to install `[docs]` and `[testing]` to verify your changes. However, `tox` will test any pull requests. You can also install `[dev-all]` to get all three of these dev dependencies.
