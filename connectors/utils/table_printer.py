from typing import List, Tuple
from tabulate import tabulate


def print_truncated_table(rows: Tuple[List[str]], columns: List[str], max_rows=6, max_columns=6):
    if len(rows) > max_rows:
        final_rows = rows[:max_rows // 2] + tuple([['...' for c in columns]]) + rows[max_rows // -2:]
    
    if len(columns) > max_columns:
        final_columns = columns[:max_columns // 2] + ['...'] + columns[max_columns // -2:]
        final_rows = [r[:max_columns // 2] + ['...'] + r[max_columns // -2:] for r in final_rows]
    else:
        final_columns = columns
    
    table = tabulate(final_rows, headers=final_columns)
    unicode_space_table = table.replace(' ', u"\u00A0")
    print(unicode_space_table)