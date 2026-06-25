import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def visualize_outliers_boxplot(df: pd.DataFrame, column: str, output_path: str) -> None:
    """
    Visualize outliers in a box plot.

    Parameters:
    - df: DataFrame
        The input DataFrame containing the data.
    - column: str
        Name of the column for which outliers are to be visualized.
    """

    plt.figure(figsize=(8, 6))
    if df[column].dtype == "object":
        le = LabelEncoder()
        df_encoded = df.copy()
        df_encoded[column] = le.fit_transform(df[column])
        plt.boxplot(df_encoded[column], vert=False)
        plt.xticks(ticks=le.transform(le.classes_), labels=le.classes_)
    else:
        plt.boxplot(df[column], vert=False)
    plt.title(f"Box Plot of {column}")
    plt.xlabel(column)
    plt.ylabel("Data")
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()


def visualize_outliers_scatterplot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    output_path: str,
) -> None:
    """
    Plot data on a graph and save the plotted image.

    Parameters:
    - df: DataFrame
        The input DataFrame containing all data points.
    - x_column: str
        Name of the column for the x-axis.
    - y_column: str
        Name of the column for the y-axis.
    - output_path: str
        Path to save the plotted image.
    """

    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_column], df[y_column], color="blue")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title("Outlier Detection")
    plt.legend()
    plt.savefig(output_path)
    plt.show()


def visualize_outliers_histogram(
    df: pd.DataFrame,
    column: str,
    output_path: str,
) -> None:
    """
    Visualize outliers in a histogram.

    Parameters:
    - df: DataFrame
        The input DataFrame containing the data.
    - column: str
        Name of the column for which outliers are to be visualized.
    """

    plt.figure(figsize=(8, 6))
    plt.hist(df[column], bins=20, color="skyblue", edgecolor="black")
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()
