import os
import re
from enum import Enum

import numpy as np
import pandas as pd
from napari_ndev import helpers
from napari_ndev import PlateMapper
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
    
def _extract_info_from_id_string(id_string: str, id_regex: dict) -> dict:
    """ Extract information from an id string using a regex dictionary. 
    For example id_string: 
    "P14-A6__2024-07-16 25x 18HIC ncoa4 FT dapi obl 01"
    and id_regex: {
        'well': '-(\w+)__',
        'HIC': r'(\d{1,3})HIC',
        'exp': r'obl (\d{2,3})'
    }
    returns {'well': 'A6', 'HIC': '18', 'exp': '01'}
    """
    id_dict = {}
    for key, regex in id_regex.items():
        match = re.search(regex, id_string)
        if match:
            id_dict[key] = match.group(1)
        else:
            id_dict[key] = None
    return id_dict


def map_tx_dict_to_df_id_col(
    tx: dict = None, 
    tx_n_well: int = None, 
    df: pd.DataFrame = None, 
    id_column: str = None,
):
    """ Map a dictionary of treatments to a dataframes id_column. 
    This should work on either a complete dataset, or as part of an iterative
    """
    if isinstance(tx_n_well, int) or tx_n_well > 0:
        plate = PlateMapper(tx_n_well)
        plate.assign_treatments(tx)
        tx_map = plate.plate_map.set_index('well_id').to_dict(orient='index')
    else:
        tx_map = tx
    
    for id, txs in tx_map.items():
        for tx, condition in txs.items():
            if tx not in df.columns:
                df[tx] = None
            df.loc[df[id_column] == id, tx] = condition
            
    return df
    
    
    

def measure_regionprops(
    label_images: Union[List[ArrayLike], ArrayLike],
    label_names: Optional[Union[List[str], str]] = None,
    intensity_images: Optional[Union[List[ArrayLike], ArrayLike]] = None,
    intensity_names: Optional[Union[List[str], str]] = None,
    properties: List[str] = ['area'],
    scale: Union[Tuple[float, float], Tuple[float, float, float]] = (1, 1),
    id_string: str = None, 
    id_regex_dict: Optional[dict] = None,
    tx_id: Optional[str] = None,
    tx_dict: Optional[dict] = None,
    tx_n_well: Optional[int] = None,
    save_data_path: PathLike = None,
) -> pd.DataFrame:
    """ Measure properties of labels with sci-kit image regionprops. 
    Optionally give a list of intensity_images to measure intensity properties 
    of labels (i.e. 'intensity_mean', 'intensity_min', 'intensity_max', 
    'intensity_std'). If no label or intensity names are given, the names are 
    automatically generated as a string of the input variable name.
    Choose from a list of properties to measure: [
            "area",
            "area_convex",
            "bbox",
            "centroid",
            "eccentricity",
            "extent",
            "feret_diameter_max",
            "intensity_max",
            "intensity_mean",
            "intensity_min",
            "intensity_std",
            "num_pixels",
            "orientation",
            "perimeter",
            "solidity",
        ]
    
    """
    from skimage import measure
    measure_dict = _generate_measure_dict(
        label_images, 
        label_names, 
        intensity_images, 
        intensity_names
    )
    
    if intensity_images is not None:
        if len(measure_dict['intensity_images']) == 1:
            intensity_stack = measure_dict['intensity_images'][0]
        else:
            intensity_stack = np.stack(measure_dict['intensity_images'], axis=-1)
    else:
        intensity_stack = None
        
    measure_props = measure.regionprops_table(
        label_image=measure_dict['label_images'][0],
        intensity_image=intensity_stack,
        properties=properties,
        spacing=scale,
    )
    
    measure_df = pd.DataFrame(measure_props)
    measure_df.insert(0, 'id_string', id_string)
    
    if id_regex_dict is not None:
        id_dict = _extract_info_from_id_string(id_string, id_regex_dict)
        for key, value in id_dict.items():
            measure_df.insert(1, key, value)
            
    if tx_id is not None and tx_dict is not None:
        map_tx_dict_to_df_id_col(tx_dict, tx_n_well, measure_df, tx_id)
            
    if save_data_path is not None:
        measure_df.to_csv(save_data_path, index=False)
    
    return measure_df
