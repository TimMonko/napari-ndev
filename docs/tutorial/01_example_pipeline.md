# Example Pipeline

## Image Utilities

1. Batch Concatenate Images and Update Metadata

![Image Utilities](screenshots/image-concatenation.png)

## Utilize Example workflow

1. Utilize simple workflow

1. DAPI: voronoi otsu (5,1)
extend_labeling_via_voronoi();

2. SYTO: median_sphere (1,1); top_hat_sphere (5,5); voronoi_otsu_labeling (1,1);

3. Membrane: median_sphere (1,1); greater_constant (200);
could also remove small objects


5. Mitochondria: median_sphere (1,1); top_hat_sphere (10,10); threshold_otsu ();
Should I do a simpler threshold otsu? Then subtract out the nuclei?

multiply_images extend_labeling and greater_constant

## Measure Widget

1. Measure objects inside each cell
