def downsample_profile(mat: Tensor, dropout: float, method="new", randsamp=False) -> Tensor:
    """
    adopted from scPRINT2
    
    This function downsamples the expression profile of a given single cell RNA matrix.

    The noise is applied based on the renoise parameter,
    the total counts of the matrix, and the number of genes. The function first calculates the noise
    threshold (scaler) based on the renoise parameter. It then generates an initial matrix count by
    applying a Poisson distribution to a random tensor scaled by the total counts and the number of genes.
    The function then models the sampling zeros by applying a Poisson distribution to a random tensor
    scaled by the noise threshold, the total counts, and the number of genes. The function also models
    the technical zeros by generating a random tensor and comparing it to the noise threshold. The final
    matrix count is calculated by subtracting the sampling zeros from the initial matrix count and
    multiplying by the technical zeros. The function ensures that the final matrix count is not less
    than zero by taking the maximum of the final matrix count and a tensor of zeros. The function
    returns the final matrix count.

    Args:
        mat (torch.Tensor): The input matrix.
        dropout (float): The renoise parameter.

    Returns:
        torch.Tensor: The matrix count after applying noise.
    """
    # Randomly drop on average N counts to each element of expression using a heavy tail Gaussian distribution
    # here we try to get the scale of the distribution so as to remove the right number of counts from each gene
    # https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02601-5#:~:text=Zero%20measurements%20in%20scRNA%2Dseq,generation%20of%20scRNA%2Dseq%20data.
    if randsamp:
        dropout = torch.rand(mat.shape[0], device=mat.device) * dropout
        dropout = (
            dropout.unsqueeze(1)
            if len(mat.shape) == 2
            else dropout.unsqueeze(1).unsqueeze(1)
        )
    if method == "old":
        totcounts = mat.sum(-1)
        ngenes = mat.shape[-1]
        tnoise = 1 - (1 - dropout) ** (1 / 2)
        # we model the sampling zeros (dropping 30% of the reads)
        res = torch.poisson(
            torch.rand(mat.shape, device=mat.device)
            * ((tnoise * totcounts.unsqueeze(-1)) / (0.5 * ngenes))
        ).int()
        # we model the technical zeros (dropping 50% of the genes)
        drop = (torch.rand(mat.shape, device=mat.device) > tnoise).int()

        mat = (mat - res) * drop
        return torch.maximum(
            mat,
            torch.zeros(
                (1, 1) if len(mat.shape) == 2 else (1, 1, 1),
                device=mat.device,
                dtype=torch.int,
            ),
        )
    elif method == "jules":
        scaler = (1 - dropout) ** (1 / 2)
        notdrop = (
            torch.rand(
                mat.shape,
                device=mat.device,
            )
            < scaler
        ).int()
        notdrop[mat == 0] = 0
        # apply the dropout after the poisson, right?
        return notdrop * torch.poisson(mat * scaler)
    elif method == "new":
        dropout = dropout * 1.1
        # we model the sampling zeros (dropping 30% of the reads)
        res = torch.poisson((mat * (dropout / 2))).int()
        # we model the technical zeros (dropping 50% of the genes)
        notdrop = (torch.rand(mat.shape, device=mat.device) >= (dropout / 2)).int()
        mat = (mat - res) * notdrop
        return torch.maximum(
            mat,
            torch.zeros(
                (1, 1) if len(mat.shape) == 2 else (1, 1, 1),
                device=mat.device,
                dtype=torch.int,
            ),
        )
    else:
        raise ValueError(f"method {method} not recognized")