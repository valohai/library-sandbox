import csv
from typing import List, Tuple

from connectors.serializers.base import BaseSerializer


class CSVSerializer(BaseSerializer):
    def serialize(self, rows: Tuple[List[str]], columns: List[str]):
        with open(self.output_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for row in rows:
                writer.writerow([str(val) for val in row])