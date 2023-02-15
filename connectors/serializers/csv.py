import csv

from connectors.serializers.base import BaseSerializer


class CSVSerializer(BaseSerializer):
    def serialize(self, rows: tuple[list[str]], columns: list[str]):
        with open(self.output_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for row in rows:
                writer.writerow([str(val) for val in row])
