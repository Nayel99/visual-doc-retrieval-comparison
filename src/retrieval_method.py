from abc import ABC, abstractmethod
from typing import List, Dict

class RetrievalMethod(ABC):
    """
    Abstract class for Retrieval Methods.
    """

    def __init__(self):
        pass

    @abstractmethod
    def query2vector(self, query: str) -> Dict[str, List]:
        """
        Convert a query to a vector.
        """
        pass

    @abstractmethod
    def image2vector(self, image_base64: str) -> List[Dict[str, List]]:
        """
        Convert an query to a vector.
        image_base64: base64 encoded image
        """
        pass
