import snowflake.connector
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

import textwrap

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
        return snowflake.connector.connect(
            user=username,
            account=account,
            private_key=passphrase,
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
