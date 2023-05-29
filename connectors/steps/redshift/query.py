import json
import os

import valohai

from connectors.steps.redshift.connect import connect
from connectors.utils.table_printer import print_truncated_table
from utils.params import parse_boolean
from utils.serializers import get_serializer


def main():
    # Redshift environment variables
    host = os.environ.get("RSHOST")
    db_name = os.environ.get("RSDATABASE")
    db_user = os.environ.get("RSUSER")
    db_password = os.environ.get("RSPASSWORD")
    cluster_identifier = os.environ.get("RSCLUSTERIDENTIFIER")
    region = os.environ.get("RSREGION")
    port = int(os.environ.get("RSPORT", 5439))
    use_iam = parse_boolean(os.environ.get("RSIAM", "1"))

    # Step parameters
    datum_alias = valohai.parameters("datum_alias").value
    output_path = valohai.parameters("output_path").value
    output_path = valohai.outputs().path(output_path if output_path else "results.csv")
    sql_query = valohai.parameters("sql_query").value

    with connect(
        use_iam=use_iam,
        host=host,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password,
        cluster_identifier=cluster_identifier,
        region=region,
        port=port,
    ) as conn:
        print(sql_query + "\n")

        with conn.cursor() as cur:
            cur.execute(sql_query)
            rows = cur.fetchall()
            columns = [i[0] for i in cur.description]

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
