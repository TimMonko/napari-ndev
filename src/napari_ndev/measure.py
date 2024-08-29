import os
import re
from enum import Enum

import numpy as np
import pandas as pd
from napari_ndev import helpers
from typing import List, Union, Optional, Tuple
from bioio_base.types import ArrayLike, PathLike

def _convert_to_list(arg: Union[List, ArrayLike, str, None]):
    """convert any non-list arguments to lists"""
    if arg is None:
        return None
    if not isinstance(arg, list):
        return [arg]
    return arg

def _generate_measure_dict(
    label_images: Union[List[ArrayLike], ArrayLike], 
    label_names: Optional[Union[List[str], str]] = None,
    intensity_images: Optional[Union[List[ArrayLike], ArrayLike]] = None,
    intensity_names: Optional[Union[List[str], str]] = None
) -> dict:
    """ Generate a dictionary of label and intensity images with their names. 
    """
    label_images = _convert_to_list(label_images)
    intensity_images = _convert_to_list(intensity_images)
    label_names = _convert_to_list(label_names)
    intensity_names = _convert_to_list(intensity_names)
    
    # automatically generate label and intensity names if not given
    if label_names is None:
        label_names = [f'label_{i}' for i in range(len(label_images))]
    if intensity_names is None and intensity_images is not None:
        intensity_names = [f'intensity_{i}' for i in range(len(intensity_images))]
        
    return {
        'label_images': label_images,
        'label_names': label_names,
        'intensity_images': intensity_images,
        'intensity_names': intensity_names,
    }

def measure_regionprops(
    label_images: Union[List[ArrayLike], ArrayLike],
    label_names: Optional[Union[List[str], str]] = None,
    intensity_images: Optional[Union[List[ArrayLike], ArrayLike]] = None,
    intensity_names: Optional[Union[List[str], str]] = None,
    filename: str = None, 
    save_data_path: PathLike = None
) -> pd.DataFrame:
    """ Measure properties of labels with sci-kit image regionprops. 
    Optionally give a list of intensity_images to measure intensity properties 
    of labels (i.e. 'intensity_mean', 'intensity_min', 'intensity_max', 
    'intensity_std'). If no label or intensity names are given, the names are 
    automatically generated as a string of the input variable name.
    
    """
    from skimage import measure
    measure_dict = _generate_measure_dict(label_images, intensity_images, label_names, intensity_names)
    