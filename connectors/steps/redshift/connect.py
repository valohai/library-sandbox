import redshift_connector
import requests
import json


def connect(
        use_iam: bool,
        host: str,
        db_name: str,
        db_user: str,
        db_password: str,
        cluster_identifier: str,
        region: str,
        port: int,
        vh_worker_role: str
    ):
    if use_iam:
        aws_metadata_ip = '169.254.169.254'
        url = f'http://{aws_metadata_ip}/latest/meta-data/iam/security-credentials/{vh_worker_role}'
        response = requests.get(url)
        credentials = json.loads(response.text)
        return redshift_connector.connect(
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
        return redshift_connector.connect(
            host=host,
            database=db_name,
            port=port,
            user=db_user,
            password=db_password,
            region=region,
        )