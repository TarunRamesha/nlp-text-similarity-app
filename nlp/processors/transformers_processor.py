import torch 
import logging
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from typing import Callable, Union, List
from transformers import AutoTokenizer, AutoModel
from nlp.processors.embedding_processor import EmbeddingProcessor
from nlp.utils.pt_helpers import mean_pooling_pt, max_pooling_pt, cls_pooling_pt

class TransformersProcessor(EmbeddingProcessor):
    """
    Provides a Transformers based implementation of the abstract class EmbeddingProcessor.
    """

    def __init__(
        self, 
        model_name_or_path: str, 
        pooling_operation: Callable, 
        logger: logging.Logger = logging.getLogger(__name__)
    ) -> None:
        """
        Initializes the TransformersProcessor with a model, tokenizer, and pooling function.
        
        Args:
            model_name_or_path (str): Path or name of the pre-trained model to load.
            pooling_operation (Callable): Function to apply pooling on token embeddings.
            logger (logging.Logger, optional): Logger instance for logging. Defaults to a module-level logger.
        """

        self._logger = logger
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model = AutoModel.from_pretrained(model_name_or_path)
        self.pooling = pooling_operation

    
    def encode(
        self, 
        corpus: Union[str, List[str]], 
        normalize_embeddings: bool = False
    ) -> torch.Tensor:
        """
        Encodes a given text corpus into embeddings using the Transformer model.
        
        Args:
            corpus (Union[str, List[str]]): A single string or a list of strings representing the text corpus.
            normalize_embeddings (bool, optional): Whether to normalize the resulting embeddings. Defaults to False.
        
        Returns:
            torch.Tensor: A tensor containing the computed embeddings of shape (num_sentences, embedding_dim).
        
        This function tokenizes the input text, processes it through the Transformer model,
        applies the defined pooling operation to extract meaningful embeddings,
        and optionally normalizes the embeddings.
        """
        
        if isinstance(corpus, str):
            corpus = [corpus]
        
        encoded_input = self._tokenizer(
            corpus, padding=True, truncation=True, return_tensors="pt"
        )
        
        with torch.no_grad():
            model_output = self._model(**encoded_input)
            model_output = self.pooling(model_output[0], encoded_input["attention_mask"])

            if normalize_embeddings:
                model_output = F.normalize(model_output, p=2, dim=1)
        
        return model_output
    
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
        return F.cosine_similarity(tensor_one, tensor_two, dim=1)

    def dot_product(self, tensor_one, tensor_two):
        """
        Computes the dot product between two tensors.

        Args:
            tensor_one (torch.Tensor): The first tensor.
            tensor_two (torch.Tensor): The second tensor.

        Returns:
            torch.Tensor: A tensor containing the dot product scores.
        """
        return torch.mm(tensor_one, tensor_two.transpose(0, 1))[0].cpu()
    
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Converts a PyTorch tensor to a NumPy array.

        Args:
            tensor (torch.Tensor): The input tensor.

        Returns:
            np.ndarray: A NumPy array representing the input tensor.
        """
        return tensor.cpu().detach().numpy()