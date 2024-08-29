import numpy as np
import pytest
from typing import Union, List
from bioio_base.types import ArrayLike
from napari_ndev.measure import _generate_measure_dict, _convert_to_list

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
