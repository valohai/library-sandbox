import json

from google.cloud import bigquery
from google.oauth2 import service_account


def connect(
    use_iam: bool,
    project: str,
    keyfile_contents: str,
):
    if use_iam:
        return bigquery.Client(project=project)

    service_account_info = json.loads(keyfile_contents)
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
    )
    return bigquery.Client(project=project, credentials=credentials)
