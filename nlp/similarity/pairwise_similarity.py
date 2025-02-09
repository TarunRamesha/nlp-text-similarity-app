import torch
import logging
from typing import List, Union
from nlp.processors.embedding_processor import EmbeddingProcessor

class PairwiseSimilarity:
    """
    A class to compute pairwise similarity between two sets of texts using a specified metric ('cosine' or 'dot')
    """

    def __init__(self, embedding_processor: EmbeddingProcessor, logger: logging.Logger = logging.getLogger(__name__)):
        """
        Initializes the PairwiseSimilarity class.
        
        Args:
            embedding_processor (EmbeddingProcessor): An object responsible for encoding text into vector embeddings.
            logger (logging.Logger, optional): Logger instance for logging information. Defaults to a module-level logger.
        """
        self._model = embedding_processor
        self.logger = logger
    
    def pairwise_similarity(self, text1: List[str], text2: List[str], metric: str ="cosine") -> List[List[float]]:
        """
        Computes the pairwise similarity between two sets of texts.
        
        Args:
            text1 (List[str]): A list of input texts.
            text2 (List[str]): A second list of input texts.
            metric (str, optional): Similarity metric to use ('cosine' or 'dot'). Defaults to "cosine".
        
        Returns:
            List[List[float]]: A matrix of similarity scores where each row corresponds to a text from text1 and each column corresponds to a text from text2.
        """
        text_vectors = self._model.encode(text1 + text2)

        if metric == "dot":
            scores = self._model.dot_product(text_vectors[:len(text1)], text_vectors[:len(text1)])
        else: # assume cosine
            to_stack = [self._model.cosine_similarity(item, text_vectors[len(text1):]) for item in text_vectors[:len(text1)]]
            scores = torch.stack(to_stack)
        
        return scores.tolist()