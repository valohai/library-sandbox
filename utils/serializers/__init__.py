from utils.serializers.csv import CSVSerializer


def get_serializer(output_path):
    if output_path.endswith(".csv"):
        return CSVSerializer(output_path)
    raise ValueError(f"Unsupported output format: {output_path}")
