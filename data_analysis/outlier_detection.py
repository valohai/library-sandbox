from enum import Enum

import pandas as pd
import valohai

from data_analysis.outlier_detection_methods import (
    detect_by_autoencoder,
    detect_by_iqr,
    detect_by_isolation_forest,
    detect_by_zscore,
)
from data_analysis.plots.visualizations import (
    visualize_outliers_boxplot,
    visualize_outliers_histogram,
    visualize_outliers_scatterplot,
)


class OutlierDetectionMethod(Enum):
    Z_SCORE = "zscore"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation-forest"
    AUTOENCODER = "autoencoder"

    def __str__(self):
        return self.value

    @classmethod
    def cast(cls, value: str):
        if isinstance(value, OutlierDetectionMethod):
            return value
        if not value:
            return None
        return OutlierDetectionMethod(str(value).lower())


def detect_outliers(
    df: pd.DataFrame,
    columns: list[str],
    method: OutlierDetectionMethod | None,
    thresholds: dict[str, float],
):
    """
    Detect outliers in a DataFrame.

    Parameters:
    - df: DataFrame
        The input DataFrame.
    - columns: list[str]
        List of columns to consider for outlier detection. If None, all columns will be considered.
    - method: OutlierDetectionMethod
        Method for outlier detection.
    - threshold: float
        Threshold value for outlier detection.

    Returns:
    - DataFrame
        DataFrame containing rows identified as outliers.
    - DataFrame
        DataFrame containing rows not identified as outliers.
    """

    if columns is None:
        columns = df.columns

    outliers = pd.DataFrame()
    non_outliers = pd.DataFrame()
    match method:
        case OutlierDetectionMethod.Z_SCORE:
            outliers, non_outliers = detect_by_zscore(
                df,
                columns=columns,
                threshold=thresholds["zscore"],
            )
        case OutlierDetectionMethod.IQR:
            outliers, non_outliers = detect_by_iqr(
                df,
                columns=columns,
                threshold=thresholds["iqr"],
            )
        case OutlierDetectionMethod.ISOLATION_FOREST:
            outliers, non_outliers = detect_by_isolation_forest(
                df,
                columns=columns,
                threshold=thresholds["isolation_forest"],
            )
        case OutlierDetectionMethod.AUTOENCODER:
            outliers, non_outliers = detect_by_autoencoder(
                df,
                columns=columns,
                percentile=thresholds["autoencoder"],
            )
    return outliers.drop_duplicates(), non_outliers.drop_duplicates()


def main():
    input_file_path = valohai.inputs("input-file").path()
    column = valohai.parameters("column").value
    thresholds = {
        "zscore": valohai.parameters("zscore_threshold").value,
        "iqr": valohai.parameters("iqr_threshold").value,
        "isolation_forest": valohai.parameters("isolation_forest_contamination").value,
        "autoencoder": valohai.parameters("autoencoder_percentile").value,
    }
    x_axis_col_name = valohai.parameters("x_axis_for_visualization").value
    output_path = valohai.parameters("output_path").value
    is_save_visualizations = valohai.parameters("save_visualizations").value

    if input_file_path and input_file_path.endswith(".csv"):
        df = pd.read_csv(input_file_path)
        all_outliers_from_all_methods = pd.DataFrame()
        for method in OutlierDetectionMethod:
            outliers, non_outliers = detect_outliers(
                df,
                columns=[column],
                method=method,
                thresholds=thresholds,
            )
            print(f"Number of outliers detected via method ({method}): {len(outliers)}")
            if len(outliers) > 0:
                all_outliers_from_all_methods = all_outliers_from_all_methods._append(
                    outliers,
                )
                outliers.to_csv(valohai.outputs().path(f"{method}/{output_path}"))

        all_outliers_from_all_methods.drop_duplicates().to_csv(
            valohai.outputs().path(f"possible_outliers_from_column_{column}.csv"),
        )
        if is_save_visualizations:
            visualize_outliers_scatterplot(
                df,
                x_column=x_axis_col_name,
                y_column=column,
                output_path=valohai.outputs().path(
                    f"scatter_plot_{column}.png",
                ),
            )

        if is_save_visualizations:
            visualize_outliers_histogram(
                df,
                column,
                output_path=valohai.outputs().path(f"histogram_plot_{column}.png"),
            )
            visualize_outliers_boxplot(
                df,
                column,
                output_path=valohai.outputs().path(f"box_plot_{column}.png"),
            )


if __name__ == "__main__":
    main()
