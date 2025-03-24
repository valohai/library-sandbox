from collections.abc import Sequence

from tabulate import tabulate


def print_truncated_table(
    rows: Sequence[Sequence[str]],
    columns: Sequence[str],
    max_rows=6,
    max_columns=6,
):
    if len(rows) > max_rows:
        final_rows = [
            *rows[: max_rows // 2],
            ["..." for c in columns],
            *rows[max_rows // -2 :],
        ]
    else:
        final_rows = list(rows)

    if len(columns) > max_columns:
        final_columns = [
            *columns[: max_columns // 2],
            "...",
            *columns[max_columns // -2 :],
        ]
        final_rows = [
            [*r[: max_columns // 2], "...", *r[max_columns // -2 :]] for r in final_rows
        ]
    else:
        final_columns = list(columns)

    table = tabulate(final_rows, headers=final_columns)
    unicode_space_table = table.replace(" ", "\u00a0")
    print(unicode_space_table)
