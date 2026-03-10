1) The algorithmic engine of scVIDR is defined in the new_vidr_model.py script. The class is called VIDRModule().

This script has the PyTorch neural network architecture.

It creates the VIDREncoder and VIDRDecoder objects, handles inference of the latent variable 'z' and generates predictions from 'z' with the decoder.

It does NOT handle AnnData objects, cell types, or anything related to biology specifically.


2) The user-facing wrapper class is called VIDR() and is defined in new_vidr.py.

Acts as a bridge between scRNAseq data and the PyTorch engine. Manages the AnnData object, sets up dataloaders, configures training loop.

It also provides the interesting biological methods for latent space arithmetic and plotting functions.


3) VIDRMODULE IS USED INSIDE OF VIDR THROUGH __INIT__.
    self.module = VIDRModule(
        n_input=self.summary_stats["n_vars"],
        n_hidden=n_hidden,
        n_latent=n_latent,)...



4) Workflow:
    - Data preparation: (VIDR)
        pass AnnData object to VIDR.setup_anndata()
    
    - Batching: (VIDR)
        - DataSplitter chunks AnnData into dictionaries of PyTorch tensors when model.train() is called.

    - Forward pass: (VIDRModule)
        - Dictionaries are passed into VIDRModule via TrainRunner. The module runs inference() step, the Encoder, to get the latent representation z. It then uses the generative() step, the Decoder, to get the reconstructions.

    - Loss calculation: (VIDRModule)
        - uses the loss() method and returns a LossOutput.

    - Optimization: (VIDR)
    
## How  the model works:
Per-Cell, Every Batch
Everything in this model operates at the individual cell level — there is no per-cell-type averaging during training.

1. Encoding: one Gaussian per cell
Each cell's log-normalized expression vector x (shape 1 × 2000 genes) passes through the encoder MLP:


x → [FC → BatchNorm → LeakyReLU → Dropout] × n_layers → hidden h
h → Linear → mean   (shape: 1 × n_latent)
h → Linear → log_var → exp() → var  (shape: 1 × n_latent)
Every cell gets its own (mean, var) pair in latent space. This is the approximate posterior q(z|x) — the model's belief about where this specific cell lives in latent space.

Then the reparameterization trick draws a sample:


z = mean + eps * sqrt(var),   eps ~ N(0,1)
This z (shape 1 × n_latent) is the actual latent point used for everything downstream that training step.

2. Decoding: deterministic from z

z → [FC → BatchNorm → LeakyReLU → Dropout] × n_layers → Linear → x_hat
x_hat is a reconstruction of the input gene expression, same shape as x.

3. Loss: three terms, all per-cell
Reconstruction (MSE):


recon_loss = ((x - x_hat) ** 2).sum(dim=1)   # shape: (n_cells,)
Summed over genes, one scalar per cell.

KL divergence — analytical, per cell:


kl_loss = KL( N(mean, var) || N(0, 1) ).sum(dim=1)   # shape: (n_cells,)
This is the per-cell KL from the cell's inferred Gaussian to the standard Normal prior. Summed over latent dimensions, one scalar per cell. Weighted by kl_weight = 0.00005 (very small — this model barely regularizes toward the prior).

NCA loss — operates on the whole batch together:


dist = torch.cdist(qz_m, qz_m)    # (batch × batch) pairwise distance matrix
p = softmax(-dist)                 # soft nearest-neighbor probabilities
For each cell i, p[i,j] is the probability that cell j is cell i's "nearest neighbor" in latent space. The loss then asks: is the probability mass concentrated on cells with the same batch (condition) AND the same cell type label?

A mask is constructed where mask[i,j] = 1/(count of j's cell type) if j has the same cell type as i, else 0. The NCA loss = sum(p * mask), which is maximized when same-type cells pull close together. Note it uses qz_m (the mean), not the sampled z — so it's acting on the clean, deterministic center of each cell's distribution.

Combined:


loss = 0.5 * recon_loss.mean() + 0.5 * (kl_loss * kl_weight).mean() - 10 * nca_loss
The NCA is subtracted (maximized) and scaled by 10.