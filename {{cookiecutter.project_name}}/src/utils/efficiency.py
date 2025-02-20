def count_model_parameters(model, trainable_only=True):
    """
    Counts the number of parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        trainable_only (bool): If True, counts only trainable parameters.

    Returns:
        int: Total number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())