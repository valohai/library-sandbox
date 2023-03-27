import csv
from collections.abc import Iterable
from typing import Any

from connectors.serializers.base import BaseSerializer


class CSVSerializer(BaseSerializer):
    def serialize(self, data: Iterable[Iterable[Any]], columns: list[str]):
        with open(self.output_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for row in data:
                writer.writerow([str(val) for val in row])
