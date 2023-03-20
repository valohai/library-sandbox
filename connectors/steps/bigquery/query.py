import json
import os

import valohai

from connectors.serializers import get_serializer
from connectors.steps.bigquery.connect import connect
from connectors.utils.table_printer import print_truncated_table


def main():
    # bigquery environment variables
    use_iam = bool(int(os.environ.get("GCP_IAM", True)))
    project = os.environ.get("GCP_PROJECT")
    keyfile_contents = os.environ.get("GCP_KEYFILE_CONTENTS_JSON")

    # Step parameters
    datum_alias = valohai.parameters("datum_alias").value
    output_path = valohai.parameters("output_path").value
    output_path = valohai.outputs().path(output_path if output_path else "results.csv")
    sql_query = valohai.parameters("sql_query").value

    with connect(
        use_iam=use_iam,
        project=project,
        keyfile_contents=keyfile_contents,
    ) as conn:
        print(sql_query + "\n")

        rows = ()
        columns = []

        query_job = conn.query(sql_query)
        result = query_job.result()
        columns = [field.name for field in result.schema]
        rows = tuple([str(cell) for cell in row] for row in result)

        serializer = get_serializer(output_path)
        serializer.serialize(rows, columns)

        if datum_alias:
            metadata = {"valohai.alias": datum_alias}
            metadata_path = valohai.outputs().path(f"{output_path}.metadata.json")
            with open(metadata_path, "w") as outfile:
                json.dump(metadata, outfile)

        if len(rows) > 0:
            print_truncated_table(rows, columns, max_rows=6, max_columns=6)
            print(f"\nTotal rows: {len(rows)}")
        else:
            print("Warning: No results for the query.")


if __name__ == "__main__":
    main()
