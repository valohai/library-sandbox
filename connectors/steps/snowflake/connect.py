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
        pkb = decrypt_key(private_key, passphrase)
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


def decrypt_key(private_key: str, passphrase: str):
    key = f"-----BEGIN ENCRYPTED PRIVATE KEY-----\n{private_key}\n-----END ENCRYPTED PRIVATE KEY-----"
    p_key = serialization.load_pem_private_key(
        key.encode(),
        password=passphrase.encode(),
        backend=default_backend(),
    )
    return p_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
