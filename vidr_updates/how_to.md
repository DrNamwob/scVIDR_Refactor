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
    

