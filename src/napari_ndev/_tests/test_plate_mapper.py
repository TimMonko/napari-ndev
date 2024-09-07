from __future__ import annotations

import pandas as pd
import pytest

from napari_ndev._plate_mapper import PlateMapper


@pytest.fixture
def plate_mapper():
    return PlateMapper(96)


@pytest.fixture
def treatments():
    return {
        'Treatment1': {'Condition1': ['A1', 'B2'], 'Condition2': ['C3']},
        'Treatment2': {'Condition3': ['D4:E5']},
    }


def test_plate_mapper_create_empty_plate_map(plate_mapper: PlateMapper):
    plate_map_df = plate_mapper.create_empty_plate_map()

    assert isinstance(plate_map_df, pd.DataFrame)
    assert len(plate_map_df) == 96
    assert len(plate_map_df.columns) == 3
    assert 'row' in plate_map_df.columns
    assert 'column' in plate_map_df.columns
    assert 'well_id' in plate_map_df.columns


def test_plate_mapper_assign_treatments(
    plate_mapper: PlateMapper, treatments: dict[str, dict[str, list[str]]]
):
    plate_map = plate_mapper.assign_treatments(treatments)

    assert isinstance(plate_map, pd.DataFrame)
    assert 'Treatment1' in plate_map.columns
    assert 'Treatment2' in plate_map.columns
    assert (
        plate_map.loc[plate_map['well_id'] == 'A1', 'Treatment1'].values[0]
        == 'Condition1'
    )
    assert (
        plate_map.loc[plate_map['well_id'] == 'B2', 'Treatment1'].values[0]
        == 'Condition1'
    )
    assert (
        plate_map.loc[plate_map['well_id'] == 'C3', 'Treatment1'].values[0]
        == 'Condition2'
    )
    assert (
        plate_map.loc[plate_map['well_id'] == 'D4', 'Treatment2'].values[0]
        == 'Condition3'
    )
    assert (
        plate_map.loc[plate_map['well_id'] == 'E5', 'Treatment2'].values[0]
        == 'Condition3'
    )
    assert (
        plate_map.loc[plate_map['well_id'] == 'E4', 'Treatment2'].values[0]
        == 'Condition3'
    )


def test_plate_mapper_get_pivoted_plate_map(
    plate_mapper: PlateMapper, treatments: dict[str, dict[str, list[str]]]
):
    plate_mapper.assign_treatments(treatments)
    plate_map_pivot = plate_mapper.get_pivoted_plate_map('Treatment1')

    assert isinstance(plate_map_pivot, pd.DataFrame)
    assert len(plate_map_pivot) == 8
    assert len(plate_map_pivot.columns) == 12
    assert 'A' in plate_map_pivot.index
    assert 'Condition1' in plate_map_pivot.values
    assert 'Condition2' in plate_map_pivot.values


def test_plate_mapper_get_styled_plate_map(
    plate_mapper: PlateMapper, treatments: dict[str, dict[str, list[str]]]
):
    plate_mapper.assign_treatments(treatments)
    plate_map_styled = plate_mapper.get_styled_plate_map('Treatment1')

    assert isinstance(plate_map_styled, pd.io.formats.style.Styler)
    assert plate_map_styled.caption == 'Treatment1 Plate Map'
