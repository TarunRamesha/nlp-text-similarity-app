import torch

def mean_pooling_pt(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean-pooled sentence embeddings from token embeddings, while considering padding tokens.
    
    This function calculates the mean of token embeddings for each sentence in the batch, using the
    attention mask to exclude padding tokens from the pooling process. The sentence embedding is computed
    by summing the token embeddings and dividing by the number of valid (non-padding) tokens.

    Parameters:
    -----------
    token_embeddings : torch.Tensor
        A tensor of shape (batch_size, seq_length, embedding_dim) containing the token embeddings for a batch of sentences.
        Each token in a sentence is represented by an embedding vector of dimension `embedding_dim`.
    
    attention_mask : torch.Tensor
        A tensor of shape (batch_size, seq_length) where 1 indicates a valid token (non-padding) and 0 indicates a padding token.
        The attention mask helps ignore padding tokens when pooling the embeddings.

    Returns:
    --------
    torch.Tensor
        A tensor of shape (batch_size, embedding_dim) containing the mean-pooled sentence embeddings.
    """

    # Check if the shapes match (batch_size and seq_length must match between token_embeddings and attention_mask)
    if token_embeddings.shape[0] != attention_mask.shape[0] or token_embeddings.shape[1] != attention_mask.shape[1]:
        raise ValueError("The shape of `attention_mask` must match the shape of `token_embeddings` in batch_size and seq_length.")
    
    # Expand the attention mask to match the shape of token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    # Sum token embeddings, considering only valid (non-padding) tokens
    sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    
    # Normalize the sum by the number of valid tokens (to avoid division by zero, clamp the denominator)
    sentence_embeddings /= torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    
    return sentence_embeddings

def max_pooling_pt(token_embeddings, attention_mask):
    pass

def cls_pooling_pt(model_output):
    pass