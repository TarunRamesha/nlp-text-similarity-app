import torch

def mean_pooling_pt(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Applies mean pooling to token embeddings using an attention mask.

    Args:
        token_embeddings (torch.Tensor): A tensor of shape (batch_size, seq_length, embedding_dim)
            representing the token embeddings.
        attention_mask (torch.Tensor): A tensor of shape (batch_size, seq_length) indicating valid tokens (1) 
            and padding tokens (0).

    Returns:
        torch.Tensor: A tensor of shape (batch_size, embedding_dim) representing sentence embeddings obtained 
        via mean pooling.
    """
    if token_embeddings.shape[0] != attention_mask.shape[0] or token_embeddings.shape[1] != attention_mask.shape[1]:
        raise ValueError("The shape of `attention_mask` must match the shape of `token_embeddings` in batch_size and seq_length.")
    
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    
    sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)    
    sentence_embeddings /= torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    
    return sentence_embeddings

def max_pooling_pt(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Applies max pooling to token embeddings using an attention mask.

    Args:
        token_embeddings (torch.Tensor): A tensor of shape (batch_size, seq_length, embedding_dim)
            representing the token embeddings.
        attention_mask (torch.Tensor): A tensor of shape (batch_size, seq_length) indicating valid tokens (1) 
            and padding tokens (0).

    Returns:
        torch.Tensor: A tensor of shape (batch_size, embedding_dim) representing sentence embeddings obtained 
        via max pooling.
    """
    if token_embeddings.shape[0] != attention_mask.shape[0] or token_embeddings.shape[1] != attention_mask.shape[1]:
        raise ValueError("The shape of `attention_mask` must match the shape of `token_embeddings` in batch_size and seq_length.")

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[input_mask_expanded == 0] = -1e9  # Mask padding tokens with a large negative value

    return torch.max(token_embeddings, dim=1)[0]

def cls_pooling_pt(model_output: torch.Tensor) -> torch.Tensor:
    """
    Extracts the [CLS] token representation from a Transformer model's output.

    Args:
        model_output (torch.Tensor): The model output containing hidden states.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, embedding_dim) representing the [CLS] token embeddings.
    """
    return model_output.last_hidden_state[:, 0]