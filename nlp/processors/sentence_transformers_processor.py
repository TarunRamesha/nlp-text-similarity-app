import os
import torch 
import logging
import numpy as np

from pathlib import Path
from typing import Callable, Union, List
from sentence_transformers import SentenceTransformer, util
from nlp.processors.embedding_processor import EmbeddingProcessor

class SentenceTransformerProcessor:
    """
    Provides a Transformers based implementation of the abstract class EmbeddingProcessor.
    """

    def __init__(
        self,
        model_name_or_path: str,
        logger: logging.Logger = logging.getLogger(__name__)
    ) -> None:
        """
        Initializes the SentenceTransformerProcessor with a specified model.

        Args:
            model_name_or_path (str): Name or path of the SentenceTransformer model to load
            logger (logging.Logger, optional): Logger instance for logging. Defaults to a module-level logger.
        """

        self._logger = logger
        self._model = SentenceTransformer(model_name_or_path)
    
    def encode(
        self, 
        corpus: Union[str, List[str]], 
        normalize_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Encodes the given text or list of texts into embeddings using the SentenceTransformer model.

        Args:
            corpus (Union[str, List[str]]): A string or list of strings to be encoded.
            normalize_embeddings (bool, optional): Whether to normalize the output embeddings. Defaults to False.

        Returns:
            torch.Tensor: A tensor containing the generated embeddings.
        """
        if isinstance(corpus, str):
            corpus = [corpus]
        return self._model.encode(
            corpus,
            convert_to_tensor=True,
            convert_to_numpy=False,
            normalize_embeddings=normalize_embeddings
        )
    
    def save_encoding(
        self, 
        corpus: Union[str, List[str]], 
        output_file: Union[str, Path], 
        normalize_embeddings: bool = False
    ) -> None:
        """
        Encodes a given text corpus and saves the embeddings to a file.
        
        Args:
            corpus (Union[str, List[str]]): A single string or a list of strings representing the text corpus.
            output_file (Union[str, Path]): The file path where the embeddings should be saved.
            normalize_embeddings (bool, optional): Whether to normalize the embeddings before saving. Defaults to False.
        """
        model_output = self.encode(corpus, normalize_embeddings)
        output_file = Path(output_file)

        try:
            torch.save(model_output, output_file)
            self._logger.info(f"Embeddings saved to {output_file}.")
        except Exception as e:
            self._logger.error(f"Failed to save embeddings to {output_file}. Error: {e}")


    def load_encoding(self, corpus_path: Union[str, Path]) -> torch.Tensor:
        """
        Load a tensor containing encoded embeddings from a given file path.

        Args:
            corpus_path (Union[str, Path]): The path to the file containing the saved embeddings.

        Returns:
            torch.Tensor: The loaded tensor containing the embeddings.
        """
        corpus_path = Path(corpus_path)
        try:
            return torch.load(corpus_path)
        except Exception as e:
            self._logger.error(f"Failed to load embeddings from {corpus_path}. Error: {e}")
            raise

    def cosine_similarity(self, tensor_one: torch.Tensor, tensor_two: torch.Tensor) -> torch.Tensor:
        """
        Computes the cosine similarity between two tensors.

        Args:
            tensor_one (torch.Tensor): The first tensor.
            tensor_two (torch.Tensor): The second tensor.

        Returns:
            torch.Tensor: A tensor containing the cosine similarity scores.
        """        
        return util.cos_sim(tensor_one, tensor_two)

    def dot_product(self, tensor_one, tensor_two):
        """
        Computes the dot product between two tensors.

        Args:
            tensor_one (torch.Tensor): The first tensor.
            tensor_two (torch.Tensor): The second tensor.

        Returns:
            torch.Tensor: A tensor containing the dot product scores.
        """
        return util.dot_score(tensor_one, tensor_two)
    
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Converts a PyTorch tensor to a NumPy array.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            np.ndarray: A NumPy array representing the input tensor.
        """
        return tensor.cpu().detach().numpy()