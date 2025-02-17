from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document

class DocumentProcessor(ABC):
    """Base class for document processors"""
    
    @abstractmethod
    def process(self, file_obj) -> List[Document]:
        pass