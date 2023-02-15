from abc import abstractmethod
from collections.abc import Iterable
from typing import Any


class BaseSerializer:
    def __init__(self, output_path):
        self.output_path = output_path

    @abstractmethod
    def serialize(self, data: Iterable[Iterable[Any]], columns: list[str]):
        pass
