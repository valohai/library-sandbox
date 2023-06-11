import snowflake.connector
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

def connect(
    username: str,
    password: str,
    private_key: str,
    passphrase: str,
    account: str,
    warehouse: str,
    database: str,
    schema: str,
):
    if private_key:
        p_key = serialization.load_pem_private_key(
            private_key.encode(),
            password=passphrase.encode(),
            backend=default_backend(),
        )
        pkb = p_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        return snowflake.connector.connect(
            user=username,
            account=account,
            private_key=pkb,
            warehouse=warehouse,
            database=database,
            schema=schema,
        )

    return snowflake.connector.connect(
        user=username,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database,
        schema=schema,
    )
