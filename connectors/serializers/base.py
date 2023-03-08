from abc import abstractmethod
from typing import List, Tuple


class BaseSerializer:
    def __init__(self, output_path):
        self.output_path = output_path
    
    @abstractmethod
    def serialize(self, data: Iterable[Iterable[Any]], columns: List[str]):
        pass