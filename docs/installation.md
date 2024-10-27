# Installation

**napari-ndev** is a pure Python package, and can be installed with [pip]:

```bash
pip install napari-ndev
```

The easiest way to get started with **napari-ndev** is to install all the optional dependencies (see note below) with:

```bash
pip install napari-ndev[all]
```

If you are unfamiliar with python or the command line, instead use the bundled app installer as demonstrated in [Beginner Setup](beginner_setup.md).

----------------------------------

## Optional Libraries

**napari-ndev** is most useful when interacting with some other napari plugins (e.g. napari-assistant) and can read additional filetypes. A few extra BSD3 compatible napari-plugins may be installed with [pip]:

```bash
pip install napari-ndev[extras]
```

**napari-ndev** can optionally use GPL-3 licensed libraries to enhance its functionality, but are not required. If you choose to install and use these optional dependencies, you must comply with the GPL-3 license terms. The main functional improvement is from some `bioio` libraries to support extra image formats, including `czi` and `lif` files. These libraries can be installed with [pip]:

```bash
pip install napari-ndev[gpl-extras]
```

In addition, you may need to install specific [`bioio` readers](https://github.com/bioio-devs/bioio) to support your specific image, such as `bioio-czi` and `bioio-lif` (included in `[gpl-extras]`) or `bioio-bioformats` (which needs conda installed).

### Development Libraries

For development use the `[dev]` optional libraries. You may also like to install `[docs]` and `[testing]` to verify your changes. However, `tox` will test any pull requests. You can also install `[dev-all]` to get all three of these dev dependencies.
