# Example Pipeline

## Image Utilities

We are going to start with the `Image Utilities` widget in order to concatenate the [CellPainting images](00_setup.md#cellpainting-images). This will show a common use of the Image Utilities plugin, wherein various file formats can be managed and saved in to a common OME-TIFF format, including channel names and physical pixel scaling.

![Image Utilities](screenshots/image-concatenation.png)

1. `Choose Directory` selects where images will be saved.
2. `Select files` individual or multiple files can be selected. Select the first 5 images (representing the 5 channels of 1 image).
3. `Metadata` dropdown. We will add in names to save the channels with, according to information that is useful. This could be the fluorophore (e.g. Hoescht 33342) or other identifying information (e.g. nuclei).

    1. `Channel Name(s)`: copy and paste `['H33342', 'conA', 'SYTO14', 'WGA_Phall', 'MitoTDR']`. The format you want to use is a list `[]` of strings `'a','b','etc.'`
    2. `Scale, ZYX`. Set Y and X to `0.656`. Z will be ignored since images are 2D.

4. `Batch Concat.` Pressing this button will iterate through all files in the folder, selecting them in groups of 5 (i.e. the number of original files selected) and then saving them with the above parameters.

### Investigate the images

If you want to investigate the raw images press `Open File(s)` this will open the original images with their known scale `(1,1,1)`. Each image will open as grayscale, and will not be layered.

Now, investigate your concatenated images. Go to `Select Files` and find the folder `ConcatenatedImages` inside the `Choose Directory` previously chosen. Select the first image and `Open File(s)`. This time, the images will be open to the scale we set `(0,0.656,0.656)` and with a default layering and pseudo-coloring. This is how all images get passed down throughout the plugin.

## Utilize Example workflow

1. Utilize simple workflow

1. DAPI: voronoi otsu (5,1)
extend_labeling_via_voronoi();

2. SYTO: median_sphere (1,1); top_hat_sphere (5,5); voronoi_otsu_labeling (1,1);
DO THIS

3. Membrane: median_sphere (1,1); greater_constant (200);
could also remove small objects

5. Mitochondria: median_sphere (1,1); top_hat_sphere (10,10); threshold_otsu ();
Should I do a simpler threshold otsu? Then subtract out the nuclei?
THEN, SUBTRACT OUT NUCLEI

multiply_images extend_labeling and greater_constant

## Measure Widget

1. Measure objects inside each cell
