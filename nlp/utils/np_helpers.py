import numpy as np

def mean_pooling_np(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    Computes mean pooling over token embeddings while ignoring padding tokens.

    This function averages the embeddings of valid tokens (where attention_mask == 1),
    ensuring that padding tokens do not contribute to the final sequence representation.

    Parameters:
    -----------
    token_embeddings : np.ndarray
        A NumPy array of shape (batch_size, seq_length, embedding_dim)
        containing token embeddings.
    
    attention_mask : np.ndarray
        A NumPy array of shape (batch_size, seq_length) with values 1 for valid tokens
        and 0 for padding tokens.

    Returns:
    --------
    mean_pooled: np.ndarray
        A NumPy array of shape (batch_size, embedding_dim) containing the mean-pooled
        embeddings for each sequence.
    """
    # Validate input dimensions
    if token_embeddings.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("Invalid input dimensions. Expected token_embeddings (3D) and attention_mask (2D).")
    
    batch_size, seq_length, embedding_dim = token_embeddings.shape
    if attention_mask.shape != (batch_size, seq_length):
        raise ValueError("Mismatched shapes: attention_mask must have shape (batch_size, seq_length).")
    
    # Expand dimensions to match the token_embeddings shape
    expanded_mask = np.expand_dims(attention_mask, axis=2)  # Shape: (batch_size, seq_length, 1)
    
    # Broadcast mask to match the shape of token_embeddings
    input_mask_expanded = np.broadcast_to(expanded_mask, token_embeddings.shape)  
    
    # Mask out padding tokens by element-wise multiplication
    masked_embeddings = token_embeddings * input_mask_expanded  

    # Sum valid token embeddings along the sequence length axis
    sum_embeddings = np.sum(masked_embeddings, axis=1)  # Shape: (batch_size, embedding_dim)

    # Compute number of valid tokens (to avoid division by zero)
    valid_token_counts = np.maximum(input_mask_expanded.sum(axis=1), 1e-9)  # Shape: (batch_size, embedding_dim)

    # Compute mean pooling
    mean_pooled = sum_embeddings / valid_token_counts
    
    # Replace NaN values (if any sequence has all padding tokens)
    mean_pooled = np.where(np.isnan(mean_pooled), 0, mean_pooled)
    
    return mean_pooled


def max_pooling_np(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    Applies max pooling over token embeddings while ignoring padding tokens.

    Parameters:
    -----------
    token_embeddings (np.ndarray): A 3D NumPy array of shape (batch_size, seq_length, embedding_dim)
                                  containing token embeddings.
    attention_mask (np.ndarray): A 2D NumPy array of shape (batch_size, seq_length)
                                 where 1 indicates valid tokens and 0 indicates padding tokens.

    Returns:
    --------
    np.ndarray: A 2D NumPy array of shape (batch_size, embedding_dim) containing max-pooled embeddings.
    """
    if token_embeddings.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("Invalid input dimensions. Expected token_embeddings (3D) and attention_mask (2D).")
    
    batch_size, seq_length, embedding_dim = token_embeddings.shape
    if attention_mask.shape != (batch_size, seq_length):
        raise ValueError("Mismatched shapes: attention_mask must have shape (batch_size, seq_length).")
    
    # Expand dimensions of attention mask to match token_embeddings shape
    expanded_mask = np.expand_dims(attention_mask, axis=2)  # Shape: (batch_size, seq_length, 1)
    input_mask_expanded = np.broadcast_to(expanded_mask, token_embeddings.shape)  # (batch_size, seq_length, embedding_dim)
    
    # Assign -inf to padding tokens to ignore them during max-pooling
    token_embeddings = np.where(input_mask_expanded == 1, token_embeddings, -np.inf)
    
    # Compute max along the sequence dimension
    max_pooled = np.max(token_embeddings, axis=1)  # Shape: (batch_size, embedding_dim)
    
    # Replace -inf values (which occur if all tokens were masked) with zeros
    max_pooled = np.where(np.isinf(max_pooled), 0, max_pooled)
    
    return max_pooled


def cosine_similarity(matrix_one: np.ndarray, matrix_two: np.ndarray) -> np.ndarray:
    """
    Computes the cosine similarity between two sets of vectors (matrices) while ignoring padding tokens.
    
    Cosine similarity is a measure of similarity between two non-zero vectors, defined as the cosine 
    of the angle between them. The result will be a matrix where each element (i, j) is the cosine 
    similarity between the i-th vector in matrix_one and the j-th vector in matrix_two.

    If the input matrices are 1D arrays, they will be expanded to 2D for processing.
    
    Parameters:
    -----------
    matrix_one : np.ndarray
        A 2D array of shape (batch_size, embedding_dim) containing token embeddings or vectors.
    matrix_two : np.ndarray
        A 2D array of shape (batch_size, embedding_dim) containing token embeddings or vectors.
        
    Returns:
    --------
    np.ndarray
        A 2D array of shape (batch_size_one, batch_size_two) containing the cosine similarity 
        values between each pair of vectors from matrix_one and matrix_two.
    """

    # Ensure that both matrices are 2D
    if matrix_one.ndim == 1:
        matrix_one = matrix_one[np.newaxis, :]
    if matrix_two.ndim == 1:
        matrix_two = matrix_two[np.newaxis, :]

    # Compute dot product between each pair of vectors (matrix_one and matrix_two)
    dot_product = np.dot(matrix_one, matrix_two.T)

    # Compute the norms of each vector in both matrices
    norm_one = np.linalg.norm(matrix_one, axis=1, keepdims=True)
    norm_two = np.linalg.norm(matrix_two, axis=1, keepdims=True)

    # Compute the cosine similarity by dividing the dot product by the outer product of norms
    return dot_product / (norm_one * norm_two.T)