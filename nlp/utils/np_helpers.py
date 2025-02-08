import numpy as np

def mean_pooling_np(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    Computes mean pooling of token embeddings using an attention mask.

    Args:
        token_embeddings (np.ndarray): A 3D array of shape (batch_size, seq_length, embedding_dim) representing token embeddings.
        attention_mask (np.ndarray): A 2D array of shape (batch_size, seq_length) indicating valid tokens (1) and padding tokens (0).

    Returns:
        np.ndarray: A 2D array of shape (batch_size, embedding_dim) representing the mean pooled embeddings.
    """

    if token_embeddings.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("Invalid input dimensions. Expected token_embeddings (3D) and attention_mask (2D).")
    
    batch_size, seq_length, embedding_dim = token_embeddings.shape
    if attention_mask.shape != (batch_size, seq_length):
        raise ValueError("Mismatched shapes: attention_mask must have shape (batch_size, seq_length).")
    
    expanded_mask = np.expand_dims(attention_mask, axis=2)  # Shape: (batch_size, seq_length, 1)
    input_mask_expanded = np.broadcast_to(expanded_mask, token_embeddings.shape)  
    
    masked_embeddings = token_embeddings * input_mask_expanded  
    sum_embeddings = np.sum(masked_embeddings, axis=1)  # Shape: (batch_size, embedding_dim)

    valid_token_counts = np.maximum(input_mask_expanded.sum(axis=1), 1e-9)  # Shape: (batch_size, embedding_dim)

    mean_pooled = sum_embeddings / valid_token_counts
    mean_pooled = np.where(np.isnan(mean_pooled), 0, mean_pooled)
    
    return mean_pooled


def max_pooling_np(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    Computes max pooling of token embeddings using an attention mask.

    Args:
        token_embeddings (np.ndarray): A 3D array of shape (batch_size, seq_length, embedding_dim) representing token embeddings.
        attention_mask (np.ndarray): A 2D array of shape (batch_size, seq_length) indicating valid tokens (1) and padding tokens (0).

    Returns:
        np.ndarray: A 2D array of shape (batch_size, embedding_dim) representing the max pooled embeddings.
    """

    if token_embeddings.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("Invalid input dimensions. Expected token_embeddings (3D) and attention_mask (2D).")
    
    batch_size, seq_length, embedding_dim = token_embeddings.shape
    if attention_mask.shape != (batch_size, seq_length):
        raise ValueError("Mismatched shapes: attention_mask must have shape (batch_size, seq_length).")
    
    expanded_mask = np.expand_dims(attention_mask, axis=2)  # Shape: (batch_size, seq_length, 1)
    input_mask_expanded = np.broadcast_to(expanded_mask, token_embeddings.shape)  # (batch_size, seq_length, embedding_dim)
    
    token_embeddings = np.where(input_mask_expanded == 1, token_embeddings, -np.inf)
    
    max_pooled = np.max(token_embeddings, axis=1)  # Shape: (batch_size, embedding_dim)
    max_pooled = np.where(np.isinf(max_pooled), 0, max_pooled)
    
    return max_pooled


def cosine_similarity(matrix_one: np.ndarray, matrix_two: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between two matrices of embeddings.

    Args:
        matrix_one (np.ndarray): A 2D array where each row is an embedding vector.
        matrix_two (np.ndarray): A 2D array where each row is an embedding vector.

    Returns:
        np.ndarray: A similarity matrix where entry (i, j) represents the cosine similarity between matrix_one[i] and matrix_two[j].
    """

    if matrix_one.ndim == 1:
        matrix_one = matrix_one[np.newaxis, :]
    if matrix_two.ndim == 1:
        matrix_two = matrix_two[np.newaxis, :]

    dot_product = np.dot(matrix_one, matrix_two.T)

    norm_one = np.linalg.norm(matrix_one, axis=1, keepdims=True)
    norm_two = np.linalg.norm(matrix_two, axis=1, keepdims=True)

    return dot_product / (norm_one * norm_two.T)