import torch
import numpy as np
from typing import List, Union
from abc import ABC, abstractmethod
from nlp.utils.np_helpers import cosine_similarity, mean_pooling_np

class EmbeddingProcessor(ABC):
    """
    Abstract base class for processing text embeddings. Provides common similarity and transformation utilities.
    """

    @abstractmethod
    def encode(self, corpus: List[str]) -> np.ndarray:
        """
        Encodes a list of text strings into an array of embeddings.

        Args:
            corpus (List[str]): A list of text documents to encode.

        Returns:
            np.ndarray: An array of shape (num_documents, embedding_dim) containing embeddings.
        """
        pass

    def cosine_similarity(self, tensor_one: np.ndarray, tensor_two: np.ndarray) -> np.ndarray:
        """
        Computes the cosine similarity between two numpy arrays.

        Args:
            tensor_one (np.ndarray): First embedding vector or matrix.
            tensor_two (np.ndarray): Second embedding vector or matrix.

        Returns:
            np.ndarray: Cosine similarity score(s) between the input tensors.
        """
        return cosine_similarity(tensor_one, tensor_two) 

    def dot_product(self, tensor_one: np.ndarray, tensor_two: np.ndarray) -> np.ndarray:
        """
        Computes the dot product between two numpy arrays.

        Args:
            tensor_one (np.ndarray): First embedding vector or matrix.
            tensor_two (np.ndarray): Second embedding vector or matrix.

        Returns:
            np.ndarray: The dot product between the input tensors.
        """
        return np.dot(tensor_one, tensor_two) 

    @staticmethod
    def to_numpy(array_or_tensor: Union[np.ndarray, List[List[float]], torch.Tensor]) -> np.ndarray:
        """
        Converts a list of lists, PyTorch tensor into a NumPy array.

        Args:
            array_or_tensor (Union[List[List[float]], torch.Tensor]):
                The input data to convert.

        Returns:
            np.ndarray: The input converted into a NumPy array.
        """
        if isinstance(array_or_tensor, np.ndarray):
            return array_or_tensor
        else:
            return np.array(array_or_tensor)