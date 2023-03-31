import json
import os

import valohai
from snowflake.connector import connect

from connectors.utils.table_printer import print_truncated_table
from utils.serializers import get_serializer


def main():
    # Snowflake environment variables
    username = os.environ.get("SNOWSQL_USER")
    password = os.environ.get("SNOWSQL_PWD")
    account = os.environ.get("SNOWSQL_ACCOUNT")
    warehouse = os.environ.get("SNOWSQL_WAREHOUSE")
    database = os.environ.get("SNOWSQL_DATABASE")
    schema = os.environ.get("SNOWSQL_SCHEMA")

    # Step parameters
    datum_alias = valohai.parameters("datum_alias").value
    output_path = valohai.parameters("output_path").value
    output_path = valohai.outputs().path(output_path if output_path else "results.csv")
    sql_query = valohai.parameters("sql_query").value

    with connect(
        user=username,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database,
        schema=schema,
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
