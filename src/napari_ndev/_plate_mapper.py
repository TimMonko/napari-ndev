import itertools
import string

import pandas as pd
import seaborn as sns


class PlateMapper:
    """
    A class for creating and manipulating plate maps.

    Attributes:
    - plate_size (int): The size of the plate (e.g., 96, 384).
    - wells (dict): A dictionary mapping plate sizes to the number of rows and
        columns.
    - plate_map (DataFrame): The plate map DataFrame with well labels.
    - plate_map_pivot (DataFrame): The wide-formatted plate map DataFrame with
        treatments as columns.

    Methods:
    - __init__(self, plate_size=96): Initializes a PlateMapper object.
    - create_empty_plate_map(self): Creates an empty plate map DataFrame for a
        given plate size.
    - assign_treatments(self, treatments): Assigns treatments to specific wells
        in a plate map.
    - get_pivoted_plate_map(self, treatment): Pivots a plate map DataFrame to
        create a wide-formatted DataFrame with a single treatment as columns.
    - get_styled_plate_map(self, treatment, palette='colorblind'): Styles a
        plate map DataFrame with different background colors for each unique
        value.
    """

    def __init__(self, plate_size=96):
        """
        Initializes a PlateMapper object.

        Parameters:
        - plate_size (int, optional): The size of the plate. Defaults to 96.
        """
        self.plate_size = plate_size
        self.wells = {
            6: (2, 3),
            12: (3, 4),
            24: (4, 6),
            48: (6, 8),
            96: (8, 12),
            384: (16, 24),
        }
        self.plate_map = self.create_empty_plate_map()
        self.plate_map_pivot = None

    def create_empty_plate_map(self):
        """
        Create an empty plate map DataFrame for a given plate size.

        Returns:
        - plate_map_df (DataFrame): The empty plate map DataFrame with well
            labels.
        """
        num_rows, num_columns = self.wells[self.plate_size]

        row_labels = list(string.ascii_uppercase)[:num_rows]
        column_labels = list(range(1, num_columns + 1))

        well_rows = row_labels * num_columns
        well_rows.sort()  # needed to sort the rows correctly
        well_columns = column_labels * num_rows

        well_labels_dict = {"Row": well_rows, "Column": well_columns}

        plate_map_df = pd.DataFrame(well_labels_dict)

        plate_map_df["Well ID"] = plate_map_df["Row"] + plate_map_df[
            "Column"
        ].astype(str)
        self.plate_map = plate_map_df
        return plate_map_df

    def assign_treatments(self, treatments):
        """
        Assigns treatments to specific wells in a plate map.

        Parameters:
        - treatments (dict): A dictionary mapping treatments to conditions and
            well ranges.

        Returns:
        - plate_map (DataFrame): The updated plate map with treatments
            assigned to specific wells.
        """
        for treatment, conditions in treatments.items():
            for condition, wells in conditions.items():
                for well in wells:
                    if ":" in well:
                        start, end = well.split(":")
                        start_row, start_col = start[0], int(start[1:])
                        end_row, end_col = end[0], int(end[1:])
                        well_condition = (
                            (self.plate_map["Row"] >= start_row)
                            & (self.plate_map["Row"] <= end_row)
                            & (self.plate_map["Column"] >= start_col)
                            & (self.plate_map["Column"] <= end_col)
                        )
                    else:
                        row, col = well[0], int(well[1:])
                        well_condition = (self.plate_map["Row"] == row) & (
                            self.plate_map["Column"] == col
                        )

                    self.plate_map.loc[well_condition, treatment] = condition
        return self.plate_map

    def get_pivoted_plate_map(self, treatment):
        """
        Pivot a plate map DataFrame to create a wide-formatted DataFrame with
            a single treatment as columns.

        Parameters:
        - treatment (str): The column name of the treatment variable in the
            plate map DataFrame.

        Returns:
        - plate_map_pivot (DataFrame): The wide-formatted plate map DataFrame
            with treatments as columns.
        """
        plate_map_pivot = self.plate_map.pivot(
            index="Row", columns="Column", values=treatment
        )
        self.plate_map_pivot = plate_map_pivot
        return plate_map_pivot

    def get_styled_plate_map(self, treatment, palette="colorblind"):
        """
        Style a plate map DataFrame with different background colors for each
        unique value.

        Parameters:
        - treatment (str): The column name of the treatment variable in the
            plate map DataFrame.
        - palette (str or list, optional): The color palette to use for
            styling. Defaults to 'colorblind'.

        Returns:
        - plate_map_styled (pandas.io.formats.style.Styler): The styled plate
            map DataFrame with different background colors for each unique
            value.
        """
        self.plate_map_pivot = self.get_pivoted_plate_map(treatment)

        unique_values = pd.unique(self.plate_map_pivot.values.flatten())
        unique_values = unique_values[pd.notna(unique_values)]

        color_palette = sns.color_palette(palette).as_hex()
        # Create an infinite iterator that cycles through the palette
        palette_cycle = itertools.cycle(color_palette)
        # Use next() to get the next color
        color_dict = {value: next(palette_cycle) for value in unique_values}

        def get_background_color(value):
            if pd.isna(value):
                return ""
            else:
                return f"background-color: {color_dict[value]}"

        plate_map_styled = (
            self.plate_map_pivot.style.applymap(get_background_color)
            .set_caption(f"{treatment} Plate Map")
            .format(lambda x: "" if pd.isna(x) else x)
        )

        return plate_map_styled
