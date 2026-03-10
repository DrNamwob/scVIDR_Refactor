import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
import numpy as np
from typing import Optional, Dict
from collections import Counter

# scVI Imports
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi import REGISTRY_KEYS

# Helper encoders/decoders
from new_modules import VIDREncoder, VIDRDecoder

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VIDRModule(BaseModuleClass):
    """
    The PyTorch Module for scVIDR.
    Responsible for the neural network architecture and loss calculation.

    Args:
        n_input (int): Number of input genes.
        n_hidden (int, optional): Number of nodes per hidden layer. Defaults to 800.
        n_latent (int, optional): Dimensionality of the latent space. Defaults to 10.
        n_layers (int, optional): Number of hidden layers for encoder and decoder. Defaults to 2.
        dropout_rate (float, optional): Dropout rate for neural networks. Defaults to 0.1.
        use_linear_decoder (bool, optional): Whether to use a linear decoder. Defaults to True.
        use_nca_loss (bool, optional): Whether to use NCA loss. Defaults to False.
        kl_weight (float, optional): Weight for KL divergence. Defaults to 0.00005.
        condition_loss (optional): Array of numeric condition values for continuous NCA loss
            (e.g. log-transformed doses). Only used when condition is numeric. Defaults to None.
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int = 800,
        n_latent: int = 10,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        use_linear_decoder: bool = True,
        use_nca_loss: bool = False,
        kl_weight: float = 0.00005,
        condition_loss=None,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.kl_weight = kl_weight
        self.use_nca_loss = use_nca_loss
        self.condition_loss = condition_loss

        # -- Encoder --
        self.encoder = VIDREncoder(
            input_dim=n_input,
            latent_dim=n_latent,
            hidden_dim=n_hidden,
            n_hidden_layers=n_layers,
            dropout_rate=dropout_rate,
        )

        # -- Decoder --
        self.nonlin_decoder = VIDRDecoder(
            input_dim=n_input,
            latent_dim=n_latent,
            hidden_dim=n_hidden,
            n_hidden_layers=n_layers,
            dropout_rate=dropout_rate,
        )

        self.lin_decoder = nn.Sequential(
            nn.Linear(n_latent, n_input),
            nn.BatchNorm1d(n_input, momentum=0.01, eps=0.001),
        )

        self.decoder = self.lin_decoder if use_linear_decoder else self.nonlin_decoder

    def _get_inference_input(self, tensors: dict) -> dict:
        """Parse the dictionary of tensors from the DataLoader."""
        return {"x": tensors[REGISTRY_KEYS.X_KEY]}

    @auto_move_data
    def inference(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        The Encoder Step: x -> z.

        Args:
            x (torch.Tensor): Input expression tensor.

        Returns:
            dict: Contains latent sample z, mean qz_m, variance qz_v, and distribution q.
        """
        mean, var, z = self.encoder(x)

        # Build distribution object for KL computation
        q = Normal(mean, var.sqrt())

        return {"z": z, "qz_m": mean, "qz_v": var, "q": q}

    def _get_generative_input(self, tensors: dict, inference_outputs: dict) -> dict:
        return {"z": inference_outputs["z"]}

    @auto_move_data
    def generative(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        The Decoder Step: z -> x_hat.

        Args:
            z (torch.Tensor): Latent representation.

        Returns:
            dict: Contains reconstructed expression px.
        """
        px = self.decoder(z)
        return {"px": px}

    def loss(
        self,
        tensors: dict,
        inference_outputs: dict,
        generative_outputs: dict,
        kl_weight: float = 1.0,
    ) -> LossOutput:
        """
        Computes the reconstruction + KL + NCA loss.

        Args:
            tensors (dict): Input tensors from DataLoader.
            inference_outputs (dict): Outputs from inference().
            generative_outputs (dict): Outputs from generative().
            kl_weight (float): Scaling factor for KL (passed by scvi trainer). Defaults to 1.0.

        Returns:
            LossOutput: scvi LossOutput object with all loss components.
        """
        x = tensors[REGISTRY_KEYS.X_KEY]
        px = generative_outputs["px"]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        q = inference_outputs["q"]

        # 1. Reconstruction Loss (MSE)
        recon_loss = ((x - px) ** 2).sum(dim=1)

        # 2. KL Divergence (analytical, against N(0,1) prior)
        kl_divergence = kl(
            q,
            Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v)),
        ).sum(dim=1)

        # 3. NCA Loss
        nca_loss_val = torch.tensor(0.0, device=x.device)
        if self.use_nca_loss:
            if np.any(self.condition_loss is not None):
                disc_labels = [tensors[REGISTRY_KEYS.LABELS_KEY]]
                cont_inds = tensors[REGISTRY_KEYS.BATCH_KEY].cpu().detach().numpy()
                cont_labels = [
                    torch.tensor(
                        [[self.condition_loss[int(i[0])]] for i in cont_inds]
                    ).to(device)
                ]
                nca_loss_val, _, _ = self.get_nca_loss(qz_m, disc_labels, cont_labels)
            else:
                disc_labels = [
                    tensors[REGISTRY_KEYS.BATCH_KEY],
                    tensors[REGISTRY_KEYS.LABELS_KEY],
                ]
                nca_loss_val, _, _ = self.get_nca_loss(qz_m, disc_labels, None)

        # Weighted combination
        loss = (0.5 * recon_loss + 0.5 * (kl_divergence * self.kl_weight)).mean() - (
            10 * nca_loss_val
        )

        return LossOutput(
            loss=loss,
            reconstruction_loss=recon_loss,
            kl_local=kl_divergence,
            extra_metrics={"nca_loss": nca_loss_val},
        )

    @torch.no_grad()
    def sample(
        self,
        tensors: dict,
        n_samples: int = 1,
    ) -> np.ndarray:
        """
        Generate samples from the posterior predictive distribution.

        Args:
            tensors (dict): Input tensors.
            n_samples (int, optional): Number of samples per cell. Defaults to 1.

        Returns:
            np.ndarray: Array of shape (n_cells, n_genes).
        """
        inference_outputs, generative_outputs = self.forward(
            tensors,
            inference_kwargs={"n_samples": n_samples},
            compute_loss=False,
        )
        px = Normal(generative_outputs["px"], 1).sample()
        return px.cpu().numpy()

    def get_nca_loss(
        self,
        z: torch.Tensor,
        disc: list,
        cont: list,
    ) -> torch.Tensor:
        """
        Computes the Neighborhood Component Analysis (NCA) loss.

        Args:
            z (torch.Tensor): Latent mean tensor (qz_m).
            disc (list): List of discrete label tensors.
            cont (list): List of continuous label tensors.

        Returns:
            tuple: (total_loss, disc_loss, cont_loss)
        """
        losses = []

        # Pairwise distance matrix in latent space
        dist = torch.cdist(z.clone(), z.clone(), p=2)
        p = dist.clone()
        p.diagonal().copy_(np.inf * torch.ones(len(p)))
        p = torch.softmax(-p, dim=1)

        # --- Discrete label loss ---
        if disc is not None:
            cells = len(disc[0])
            masks = np.zeros((cells, cells))
            maxVal = 0
            for Y in disc:
                Y = Y.cpu().detach().numpy()
                Y = np.asarray([y[0] for y in Y])
                counts = Counter(Y)
                area_counts = {k: v for k, v in counts.items()}
                m = Y[:, np.newaxis] == Y[np.newaxis, :]
                for k, v in area_counts.items():
                    if "nan" != str(k):
                        maxVal += 1
                        masks = masks + m * (Y == k) * (1 / v)
            if maxVal != 0:
                masks = masks * (1 / maxVal)

            masks = torch.from_numpy(masks).float().to(device)
            masked_p = p * masks
            losses.append(torch.sum(masked_p))
        else:
            losses.append(torch.tensor(0.0, device=device))

        # --- Continuous label loss ---
        if cont is not None:
            cells = len(cont[0])
            n_cont_weights = len(cont)

            weights = torch.empty(
                (cells, cells * n_cont_weights), dtype=torch.float, device=device
            )
            for n in range(n_cont_weights):
                Y = cont[n]
                cont_dists = torch.cdist(Y.clone(), Y.clone())
                cont_dists.diagonal().copy_(np.inf * torch.ones(len(cont_dists)))
                cont_dists = torch.nan_to_num(cont_dists, nan=np.inf)
                cont_dists = torch.softmax(-cont_dists, dim=1)
                weights[:, cells * n : cells * (n + 1)] = cont_dists

            for n in range(n_cont_weights):
                s, e = cells * n, cells * (n + 1)
                weight_calc = p * weights[:, s:e]
                m, _ = torch.max(weights[:, s:e], dim=1)
                masked_p = weight_calc / torch.sum(m)
                losses.append(torch.sum(masked_p))

        lossVals = torch.stack(losses, dim=0)
        disc_loss = lossVals[0]
        cont_loss = (
            torch.sum(lossVals[1:])
            if cont is not None
            else torch.tensor(0.0, device=device)
        )
        total_loss = disc_loss + cont_loss

        return total_loss, disc_loss, cont_loss
