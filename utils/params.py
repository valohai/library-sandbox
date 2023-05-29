from typing import Any

TRUE_VALUES = {"1", "true", "on", "yes", "y"}


def parse_boolean(parameter_value: Any) -> bool:
    if not parameter_value:
        return False
    return str(parameter_value).strip().lower() in TRUE_VALUES
