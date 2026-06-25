import pandas as pd
import valohai


def main():
    rm_duplicates = valohai.parameters("remove_duplicate_rows").value
    rm_null = valohai.parameters("remove_null").value
    null_look_up_cols = valohai.parameters("null_lookup_columns").value
    duplicate_lookup_cols = valohai.parameters("duplicate_lookup_columns").value
    clear_fmt = valohai.parameters("clear_formatting").value
    output_path = valohai.parameters("output_file_name").value

    df = pd.read_csv(valohai.inputs("input-file").path())

    print("null_look_up_cols:", null_look_up_cols)
    print("duplicate_lookup_cols:", duplicate_lookup_cols)
    # Remove records if missing data in particular columns are found
    null_look_up_cols = (
        null_look_up_cols.split(",")
        if isinstance(null_look_up_cols, str) and len(null_look_up_cols) > 0
        else None
    )
    duplicate_lookup_cols = (
        duplicate_lookup_cols.split(",")
        if isinstance(duplicate_lookup_cols, str) and len(duplicate_lookup_cols) > 0
        else None
    )
    print("null_look_up_cols:", null_look_up_cols)
    print("duplicate_lookup_cols:", duplicate_lookup_cols)

    if rm_null:
        is_empty_values = df.isnull().values.any()
        num__empty_values = df.isnull().sum()
        print(f"Missing values found: {is_empty_values}")
        print(f"Number of Missing values found: {num__empty_values}")

        if is_empty_values:
            print(
                f"Dropping records where following columns are null: {null_look_up_cols if null_look_up_cols else 'all'} ",
            )
            df.dropna(
                subset=null_look_up_cols if null_look_up_cols else None,
                inplace=True,
            )

    # Remove duplicate records
    if rm_duplicates:
        duplicated_df = df.duplicated(
            subset=duplicate_lookup_cols if duplicate_lookup_cols else None,
            keep="first",
        )
        is_duplicate = duplicated_df.values.any()
        num_duplicates = duplicated_df.sum()
        print(f"Duplicate records found: {is_duplicate}")
        print(f"Number of Duplicate records found: {num_duplicates}")
        if is_duplicate:
            print(
                f"Dropping duplicate records based on following columns: {duplicate_lookup_cols if duplicate_lookup_cols else 'all'}",
            )
            df.drop_duplicates(subset=duplicate_lookup_cols, inplace=True)

    # Clear formatting
    if clear_fmt:
        print("Clear formatting")
        df = df.map(lambda x: x.strip() if type(x) == str else x)

    if not output_path:
        output_path = "cleaned_data.csv"

    df.to_csv(valohai.outputs().path(output_path), index=False)


if __name__ == "__main__":
    main()
