import numpy as np
import pandas as pd
import pytest
from typing import Union, List
from bioio_base.types import ArrayLike
from napari_ndev.measure import (
    _generate_measure_dict, _convert_to_list, _extract_info_from_id_string,
    measure_regionprops
)

@pytest.mark.parametrize(
    "arg, expected",
    [
        (None, None),
        (1, [1]),
        ("string", ["string"]),
        ([1, 2, 3], [1, 2, 3]),
        (np.array([1, 2, 3]), [np.array([1, 2, 3])]),
        ([np.array([1, 2]), np.array([3, 4])], [np.array([1, 2]), np.array([3, 4])]),
    ]
)
def test_convert_to_list(arg: Union[List, ArrayLike, str, None], expected: Union[List, None]):
    result = _convert_to_list(arg)
    if isinstance(result, list):
        for res, exp in zip(result, expected):
            if isinstance(res, np.ndarray):
                assert np.array_equal(res, exp)
            else:
                assert res == exp
    else:
        assert result == expected


label1 = np.array([[1, 2], [3, 4]])
label2 = np.array([[5, 6], [7, 8]])
label_list = [label1, label2]
intensity1 = np.array([[5, 6], [7, 8]])
intensity2 = np.array([[9, 10], [11, 12]])
intensity_list = [intensity1, intensity2]

@pytest.mark.parametrize(
    "label_images, label_names, intensity_images, intensity_names, expected",
    [
        (label1, None, None, None, {
            'label_images': label1,
            'label_names': ['label_0'],
            'intensity_images': None,
            'intensity_names': None
        }),
        (label1, None, intensity1, None, {
            'label_images': label1,
            'label_names': ['label_0'],
            'intensity_images': intensity1,
            'intensity_names': ['intensity_0']
        }),
        (label1, "custom_label", intensity1, "custom_intensity", {
            'label_images': label1,
            'label_names': ["custom_label"],
            'intensity_images': intensity1,
            'intensity_names': ["custom_intensity"]
        }),
        (label1, "custom_label", None, None, {
            'label_images': label1,
            'label_names': ["custom_label"],
            'intensity_images': None,
            'intensity_names': None
        }),
        (label_list, ["label_c1", "label_c2"], intensity_list, ["intensity_c1", "intensity_c2"], {
            'label_images': label_list,
            'label_names': ["label_c1", "label_c2"],
            'intensity_images': intensity_list,
            'intensity_names': ["intensity_c1", "intensity_c2"]
        }),
    ]
)
def test_generate_measure_dict(label_images, label_names, intensity_images, intensity_names, expected):
    result = _generate_measure_dict(label_images, label_names, intensity_images, intensity_names)
    assert result['label_images'][0].all() == expected['label_images'][0].all()
    assert result['label_names'] == expected['label_names']
    assert isinstance(result['label_names'], list)
    # assert intensity images are the same, if they exist
    if result['intensity_images'] is not None:
        assert result['intensity_images'][0].all() == expected['intensity_images'][0].all()
        assert result['intensity_names'] == expected['intensity_names']
    else:
        assert result['intensity_images'] is None
        assert result['intensity_names'] is None
        
        
def test_extract_info_from_id_string():
    id_string = "P14-A6__2024-07-16 25x 18HIC ncoa4 FT dapi obl 01"
    id_regex = {
        'scene': r'(P\d{1,3}-\w+)__',
        'well': r'-(\w+)__',
        'HIC': r'(\d{1,3})HIC',
        'exp': r'obl (\d{2,3})'
    }
    expected = {'scene': 'P14-A6', 'well': 'A6', 'HIC': '18', 'exp': '01'}
    result = _extract_info_from_id_string(id_string, id_regex)
    assert result == expected

# Define test data
label_image_2d = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 0]])
label_image_3d = np.array([[[0, 0, 1], [0, 1, 1], [1, 1, 0]], [[0, 0, 1], [0, 1, 1], [1, 1, 0]]])
intensity_image_2d = np.array([[10, 10, 20], [10, 20, 20], [20, 20, 10]])
intensity_image_3d = np.array([[[10, 10, 20], [10, 20, 20], [20, 20, 10]], [[10, 10, 20], [10, 20, 20], [20, 20, 10]]])
id_string = 'P14-A6__2024-07-16 25x 18HIC ncoa4 FT dapi obl 01'
id_regex = {
    'well': '-(\w+)__',
    'HIC': r'(\d{1,3})HIC',
    'exp': r'obl (\d{2,3})'
}


@pytest.mark.parametrize(
    "label_images, label_names, intensity_images, intensity_names, properties, scale, id_string, id_regex, save_data_path, expected_columns",
    [
        (label_image_2d, None, None, None, ['area'], (1.0, 1.0), id_string, id_regex, None, ['id_string', 'well', 'HIC', 'exp', 'area']),
        ([label_image_2d], None, [intensity_image_2d], None, ['area', 'intensity_mean'], (1.0, 1.0), "test_id", None, None, ['id_string', 'area', 'intensity_mean']),
        ([label_image_3d], None, None, None, ['area'], (1.0, 1.0, 1.0), "test_id", None, None, ['id_string', 'area']),
        ([label_image_3d], None, [intensity_image_3d], None, ['area', 'intensity_mean'], (1.0, 1.0, 1.0), "test_id", None, None, ['id_string', 'area', 'intensity_mean']),
    ]
)
def test_measure_regionprops(label_images, label_names, intensity_images, intensity_names, properties, scale, id_string, id_regex, save_data_path, expected_columns):
    result_df = measure_regionprops(
        label_images=label_images,
        label_names=label_names,
        intensity_images=intensity_images,
        intensity_names=intensity_names,
        properties=properties,
        scale=scale,
        id_string=id_string,
        id_regex=id_regex,
        save_data_path=save_data_path,
    )
    
    assert isinstance(result_df, pd.DataFrame)
    # Check if the DataFrame contains the expected columns
    assert all(column in result_df.columns for column in expected_columns)
    # Check if the id_string column contains the correct value
    assert (result_df['id_string'] == id_string).all()