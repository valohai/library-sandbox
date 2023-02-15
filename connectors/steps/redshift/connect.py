import redshift_connector
import requests
from requests.exceptions import ReadTimeout


def connect(
    use_iam: bool,
    host: str,
    db_name: str,
    db_user: str,
    db_password: str,
    cluster_identifier: str,
    region: str,
    port: int,
):
    if use_iam:
        credentials = get_aws_credentials()
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

    return redshift_connector.connect(
        host=host,
        database=db_name,
        port=port,
        user=db_user,
        password=db_password,
        region=region,
        cluster_identifier=cluster_identifier,
    )


def get_imds_auth_headers(session):
    try:
        token_resp = session.put(
            url="http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            timeout=5,
        )
    except ReadTimeout:  # IMDSv1
        return {}
    if token_resp.status_code in (403, 404, 405):  # IMDSv1
        return {}
    if token_resp.status_code == 200:
        return {"X-aws-ec2-metadata-token": token_resp.text}
    token_resp.raise_for_status()
    return None


def get_aws_credentials():
    with requests.Session() as session:
        req_headers = get_imds_auth_headers(session)
        role_resp = session.get(
            url="http://169.254.169.254/latest/meta-data/iam/security-credentials/",
            headers=req_headers,
            timeout=5,
        )
        role_resp.raise_for_status()
        role_name = role_resp.text
        credentials_resp = session.get(
            url=f"http://169.254.169.254/latest/meta-data/iam/security-credentials/{role_name}",
            headers=req_headers,
            timeout=5,
        )
        credentials_resp.raise_for_status()
        return credentials_resp.json()
