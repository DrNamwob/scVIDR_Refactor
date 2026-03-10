from typing import Callable

import torch
from torch import nn
from torch.distributions import Normal


class VIDREncoder(nn.Module):
    """Variational Encoder for scVIDR (Single-Cell Variational Inference for Dose Response).

    Encodes input gene expression data into a latent representation by learning
    the mean and log-variance of a Normal distribution, then sampling via the
    reparameterization trick.

    Args:
        input_dim (int): Dimensionality of the input data (number of genes).
        latent_dim (int): Dimensionality of the latent representation.
        hidden_dim (int): Number of hidden units in each fully connected layer.
        n_hidden_layers (int): Total number of hidden layers (including input projection).
        momentum (float, optional): Momentum for BatchNorm1d. Defaults to 0.01.
        eps (float, optional): Epsilon for BatchNorm1d numerical stability. Defaults to 0.001.
        dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.2.
        reparam_eps (float, optional): Small constant added to variance before reparameterization. Defaults to 1e-4.
        nonlin (nn.Module, optional): Non-linear activation class. Defaults to nn.LeakyReLU.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        momentum: float = 0.01,
        eps: float = 0.001,
        dropout_rate: float = 0.2,
        reparam_eps: float = 1e-4,
        nonlin: Callable = nn.LeakyReLU,
    ):
        super().__init__()
        self.reparam_eps = reparam_eps

        def _fc_block(in_dim: int, out_dim: int) -> nn.Sequential:
            """Build a single FC block: Linear -> BatchNorm -> Activation -> Dropout."""
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim, momentum=momentum, eps=eps),
                nonlin(),
                nn.Dropout(p=dropout_rate),
            )

        # Input projection layer
        modules = [_fc_block(input_dim, hidden_dim)]

        # Additional hidden layers — each instantiated separately to avoid weight sharing
        for _ in range(n_hidden_layers - 1):
            modules.append(_fc_block(hidden_dim, hidden_dim))

        self.encoder = nn.Sequential(*modules)

        # Output heads for latent distribution parameters
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, inputs: torch.Tensor):
        """Encode inputs into a latent representation.

        Args:
            inputs (torch.Tensor): Input expression tensor of shape (n_cells, input_dim).

        Returns:
            tuple:
                - mean (torch.Tensor): Mean of the latent distribution, shape (n_cells, latent_dim).
                - var (torch.Tensor): Variance of the latent distribution, shape (n_cells, latent_dim).
                - latent_rep (torch.Tensor): Reparameterized latent sample, shape (n_cells, latent_dim).
        """
        h = self.encoder(inputs)
        mean = self.mean(h)
        log_var = self.log_var(h)

        # Exponentiate log-variance and add epsilon for numerical stability
        var = torch.exp(log_var) + self.reparam_eps

        # Reparameterization trick: z = mean + eps * std
        latent_rep = Normal(mean, var.sqrt()).rsample()

        return mean, var, latent_rep


class VIDRDecoder(nn.Module):
    """Variational Decoder for scVIDR (Single-Cell Variational Inference for Dose Response).

    Decodes a latent representation back into gene expression space.

    Args:
        input_dim (int): Dimensionality of the output data (number of genes).
        latent_dim (int): Dimensionality of the latent representation.
        hidden_dim (int): Number of hidden units in each fully connected layer.
        n_hidden_layers (int): Total number of hidden layers (including latent projection).
        momentum (float, optional): Momentum for BatchNorm1d. Defaults to 0.01.
        eps (float, optional): Epsilon for BatchNorm1d numerical stability. Defaults to 0.001.
        dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.2.
        nonlin (nn.Module, optional): Non-linear activation class. Defaults to nn.LeakyReLU.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        momentum: float = 0.01,
        eps: float = 0.001,
        dropout_rate: float = 0.2,
        nonlin: Callable = nn.LeakyReLU,
    ):
        super().__init__()

        def _fc_block(in_dim: int, out_dim: int) -> nn.Sequential:
            """Build a single FC block: Linear -> BatchNorm -> Activation -> Dropout."""
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim, momentum=momentum, eps=eps),
                nonlin(),
                nn.Dropout(p=dropout_rate),
            )

        # Latent projection layer
        modules = [_fc_block(latent_dim, hidden_dim)]

        # Additional hidden layers — each instantiated separately to avoid weight sharing
        for _ in range(n_hidden_layers - 1):
            modules.append(_fc_block(hidden_dim, hidden_dim))

        # Final output projection — no activation, raw reconstruction
        modules.append(nn.Linear(hidden_dim, input_dim))

        self.decoder = nn.Sequential(*modules)

    def forward(self, latent_rep: torch.Tensor) -> torch.Tensor:
        """Decode a latent representation back into expression space.

        Args:
            latent_rep (torch.Tensor): Latent tensor of shape (n_cells, latent_dim).

        Returns:
            torch.Tensor: Reconstructed expression tensor of shape (n_cells, input_dim).
        """
        return self.decoder(latent_rep)
