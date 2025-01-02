# Installation

If you are unfamiliar with python or the command line, instead use the bundled app installer as demonstrated in [Beginner Setup](beginner_setup.md).

## Install with uv

uv is the newest and fastest way to manage python libraries. It is very easy to install, and simplifies environment manage, but requires some minimal input to the command line.  [Install uv from here](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1). Then, the simplest way to install `napari-ndev`:

```bash
uv tool install napari-ndev
```

Alternatively, download the full opinionated package, which includes non-BSD3 licensed libraries with:

```bash
uv tool install napari-ndev[all]
```

Then, you can easily open napari with the command line every time by just typing:

```bash
napari-ndev
```

The tool version of `napari-ndev` effectively runs as an alias for `napari -w napari-ndev` and opens the `nDev App` upon launch. With this method, additional plugins installed via the plugin menu persist between each call to `napari-ndev`

To update a tool with uv:

```bash
uv tool upgrade napari-ndev
```

## Install with pip

**napari-ndev** is a pure Python package, and can be installed with [pip] (and it is recommended to do so in a [managed environment](https://biapol.github.io/blog/mara_lampert/getting_started_with_mambaforge_and_python/readme.html)):

```bash
pip install napari-ndev
```

The easiest way to get started with **napari-ndev** is to install all the optional dependencies (see note below) with:

```bash
pip install napari-ndev[all]
```

Afterwards, you can call from the command line (in the same environment) `napari-ndev` to open napari with the `nDev App` open on launch.

### Optional Libraries

**napari-ndev** is most useful when interacting with some other napari plugins (e.g. napari-assistant) and can read additional filetypes. A few extra BSD3 compatible napari-plugins may be installed with [pip]:

```bash
pip install napari-ndev[extras]
```

**napari-ndev** can optionally use GPL-3 licensed libraries to enhance its functionality, but are not required. If you choose to install and use these optional dependencies, you must comply with the GPL-3 license terms. The main functional improvement is from some `bioio` libraries to support extra image formats, including `czi` and `lif` files. These libraries can be installed with [pip]:

```bash
pip install napari-ndev[gpl-extras]
```

In addition, you may need to install specific [`bioio` readers](https://github.com/bioio-devs/bioio) to support your specific image, such as `bioio-czi` and `bioio-lif` (included in `[gpl-extras]`) or `bioio-bioformats` (which needs conda installed).

## Development Libraries

For development use the `[dev]` optional libraries to verify your changes, which includes the `[docs]` and `[testing]` optional groups. However, the Github-CI will test pull requests with `[testing]` only.

### Development with uv

uv can be a useful tool for building as similar an environment as possible across systems. To do so, navigate in your terminal to the `napari-ndev` source directory. `--python` sets the minimum python version. `--no-workspace` prevents discovering parent workspaces. Then:

```bash
uv init --python 3.11 --no-workspace
uv sync
```

You may use uv to set a certain python version, e.g.:

```bash
uv pin python 3.11
```

To use uv to install extras (like with `napari-ndev[dev]`), use:

```bash
uv sync --extra dev
```

You may also test the tool version of uv during development with:

```bash
uv install tool .
```

You can also test with tox in parallel ([via tox-uv](https://github.com/tox-dev/tox-uv)) with:

```bash
tox - p auto
```
