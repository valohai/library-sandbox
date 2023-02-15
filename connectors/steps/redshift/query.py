import json
from connectors.serializers import get_serializer
from connectors.serializers.base import BaseSerializer
from connectors.utils.table_printer import print_truncated_table
import redshift_connector
import valohai
import requests
import csv
import os


# Redshift environment variables
host = os.environ.get('REDSHIFT_HOST')
db_name = os.environ.get('REDSHIFT_DATABASE')
db_user = os.environ.get('REDSHIFT_USER')
db_password = os.environ.get('REDSHIFT_PASSWORD')
cluster_identifier = os.environ.get('REDSHIFT_CLUSTER_IDENTIFIER')
region = os.environ.get('REDSHIFT_REGION')
port = int(os.environ.get('REDSHIFT_PORT', 5439))
use_iam = bool(os.environ.get('REDSHIFT_USE_IAM', False))

# Other environment variables
aws_metadata_ip = os.environ.get('AWS_METADATA_IP', '169.254.169.254')
vh_worker_role = os.environ.get('VH_WORKER_ROLE', 'ValohaiWorkerRole')

# Step parameters
datum_alias = valohai.parameters('datum_alias').value
output_path = valohai.parameters('output_name').value
output_path = valohai.outputs().path(output_path if output_path else 'results.csv')
sql_query = valohai.parameters('SQLquery').value

if use_iam:
    url = f'http://{aws_metadata_ip}/latest/meta-data/iam/security-credentials/{vh_worker_role}'
    response = requests.get(url)
    credentials = json.loads(response.text)
    conn = redshift_connector.connect(
        iam=True,
        host=host,
        database=db_name,
        db_user=db_user,
        port=port,
        cluster_identifier=cluster_identifier,
        access_key_id=credentials["AccessKeyId"],
        secret_access_key=credentials["SecretAccessKey"],
        session_token=credentials["Token"],
        region=region,
    )
else:
    conn = redshift_connector.connect(
        host=host,
        database=db_name,
        port=port,
        user=db_user,
        password=db_password,
        region=region,
    )

try:
    print(sql_query + "\n")
    cur = conn.cursor()
    cur.execute(sql_query)
    rows = cur.fetchall()
    columns = [i[0] for i in cur.description]

    serializer = get_serializer(output_path)
    serializer.serialize(rows, columns)

    if datum_alias:
        metadata = {"valohai.alias": datum_alias}
        metadata_path = valohai.outputs().path(f"{output_path}.metadata.json")
        with open(metadata_path, 'w') as outfile:
            json.dump(metadata, outfile)
    
    if len(rows) > 0:
        print_truncated_table(rows, columns, max_rows=6, max_columns=6)
        print(f"\nTotal rows: {str(len(rows))}")
    else:
        print("Warning: No results for the query.")

finally:
    cur.close()
    conn.close()
