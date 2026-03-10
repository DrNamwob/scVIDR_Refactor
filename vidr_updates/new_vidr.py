import logging
from typing import Dict, List, Optional, Union

import numpy
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from adjustText import adjust_text
from anndata import AnnData
from matplotlib import pyplot
from scipy import sparse, stats
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalObsField,
    LayerField,
    NumericalObsField,
)
from scvi.dataloaders import DataSplitter

# scVI Imports
from scvi.model.base import BaseModelClass
from scvi.train import TrainingPlan, TrainRunner
from sklearn.linear_model import LinearRegression
from new_utils import random_sample

# Local imports
from vidr_model import VIDRModule

logger = logging.getLogger(__name__)

# Custom registry key for numeric condition values (e.g. dose, severity score)
CONDITION_REGISTRY_KEY = "condition_numeric"


class VIDR(BaseModelClass):
    """
    scVIDR: Variational Inference for Dose Response.

    A VAE-based model for perturbation prediction using latent space arithmetic.
    Conditions can be numeric (e.g. drug doses) or categorical (e.g. healthy vs. diseased).

    Parameters
    ----------
    adata
        AnnData object registered via :meth:`~VIDR.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers for encoder and decoder.
    dropout_rate
        Dropout rate for neural networks.
    kl_weight
        Weight for KL divergence term in loss.
    use_linear_decoder
        If True, use a linear decoder (more interpretable).
    use_nca_loss
        If True, apply NCA loss during training.
    use_condition_loss
        If True, incorporate numeric condition values into NCA loss.
        Only meaningful when condition is numeric (e.g. dose).
        Requires condition_key to be passed to setup_anndata().
    **model_kwargs
        Additional keyword arguments passed to VIDRModule.

    Examples
    --------
    >>> # Categorical condition (e.g. healthy vs. diseased)
    >>> VIDR.setup_anndata(adata, batch_key="disease_state", labels_key="cell_type")
    >>> model = VIDR(adata)
    >>> model.train()
    >>> pred, delta = model.predict(ctrl_key="healthy", treat_key="diseased", cell_type_to_predict="hepatocyte")
    >>>
    >>> # Numeric condition (e.g. drug dose)
    >>> VIDR.setup_anndata(adata, batch_key="dose_str", labels_key="cell_type", condition_key="dose_numeric")
    >>> model = VIDR(adata, use_condition_loss=True)
    >>> model.train()
    >>> pred, delta = model.predict(ctrl_key="0", treat_key="30", cell_type_to_predict="hepatocyte", continuous=True, doses=[0,1,10,30])
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 800,
        n_latent: int = 100,
        n_layers: int = 2,
        dropout_rate: float = 0.2,
        kl_weight: float = 5e-5,
        use_linear_decoder: bool = False,
        use_nca_loss: bool = False,
        use_condition_loss: bool = False,
        **model_kwargs,
    ):
        super().__init__(adata)

        # Build condition_loss array from registered numeric obs field if requested
        condition_loss = None
        if use_condition_loss:
            try:
                condition_vals = self.adata_manager.get_from_registry(
                    CONDITION_REGISTRY_KEY
                )
                condition_loss = np.log1p(condition_vals.astype(float).flatten())
            except Exception:
                logger.warning(
                    "use_condition_loss=True but condition_key not found in registry. "
                    "Make sure to pass condition_key to setup_anndata(). Ignoring condition loss."
                )

        self.module = VIDRModule(
            n_input=self.summary_stats["n_vars"],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            kl_weight=kl_weight,
            use_linear_decoder=use_linear_decoder,
            use_nca_loss=use_nca_loss,
            condition_loss=condition_loss,
            **model_kwargs,
        )

        self._model_summary_string = (
            f"VIDR Model\n"
            f"  n_hidden:           {n_hidden}\n"
            f"  n_latent:           {n_latent}\n"
            f"  n_layers:           {n_layers}\n"
            f"  dropout_rate:       {dropout_rate}\n"
            f"  linear_decoder:     {use_linear_decoder}\n"
            f"  nca_loss:           {use_nca_loss}\n"
            f"  condition_loss:     {use_condition_loss}"
        )

        self.init_params_ = self._get_init_params(locals())

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        condition_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Register AnnData fields for use with VIDR.

        Parameters
        ----------
        adata
            AnnData object to register.
        layer
            Key in adata.layers to use as input. If None, uses adata.X.
        batch_key
            Key in adata.obs for the condition used to split groups
            (e.g. "disease_state" with values "healthy"/"diseased",
            or "dose_str" with values "0"/"30"). Used as ctrl_key/treat_key in predict().
        labels_key
            Key in adata.obs for cell type labels.
        condition_key
            Key in adata.obs for a *numeric* condition value (e.g. actual dose as float).
            Only needed when use_condition_loss=True for continuous NCA loss.
            For categorical conditions (healthy/diseased), leave as None.
        """
        setup_method_args = cls._get_setup_method_args(**locals())

        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
        ]

        if condition_key is not None:
            anndata_fields.append(
                NumericalObsField(CONDITION_REGISTRY_KEY, condition_key)
            )

        adata_manager = AnnDataManager(
            fields=anndata_fields,
            setup_method_args=setup_method_args,
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        max_epochs: int = 100,
        batch_size: int = 128,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        early_stopping: bool = True,
        early_stopping_patience: int = 25,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """
        Train the VIDR model.

        Parameters
        ----------
        max_epochs
            Maximum number of training epochs.
        batch_size
            Mini-batch size.
        train_size
            Fraction of data to use for training (default 0.9).
        validation_size
            Fraction of data to use for validation. If None, uses 1 - train_size.
        early_stopping
            Whether to use early stopping.
        early_stopping_patience
            Number of epochs with no improvement before stopping.
        plan_kwargs
            Keyword arguments passed to TrainingPlan.
        **trainer_kwargs
            Additional arguments passed to TrainRunner.
        """
        plan_kwargs = plan_kwargs or {}

        data_splitter = DataSplitter(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
        )

        training_plan = TrainingPlan(self.module, **plan_kwargs)

        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            **trainer_kwargs,
        )
        return runner()

    # ------------------------------------------------------------------
    # Latent Representation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[List[int]] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return the latent representation z for all cells.

        Parameters
        ----------
        adata
            AnnData to encode. If None, uses the registered adata.
        indices
            Optional subset of cell indices.
        batch_size
            Mini-batch size for inference.

        Returns
        -------
        np.ndarray of shape (n_cells, n_latent)
        """
        self.module.eval()
        dl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        latent = []
        for tensors in dl:
            x = tensors[REGISTRY_KEYS.X_KEY].to(self.device)
            inference_outputs = self.module.inference(x)
            latent.append(inference_outputs["z"].cpu().numpy())

        return np.concatenate(latent, axis=0)

    # ------------------------------------------------------------------
    # Prediction (Latent Space Arithmetic)
    # ------------------------------------------------------------------

    def predict(
        self,
        ctrl_key: str = None,
        treat_key: str = None,
        cell_type_to_predict: str = None,
        regression: bool = False,
        continuous: bool = False,
        low_dose: bool = False,
        doses: Optional[list] = None,
    ) -> Union[AnnData, Dict]:
        """
        Predict perturbed cell states using latent space arithmetic.

        Parameters
        ----------
        ctrl_key
            Value in the batch/treatment obs column representing control cells.
        treat_key
            Value in the batch/treatment obs column representing treated cells.
        cell_type_to_predict
            The cell type to predict the perturbed state for.
        regression
            If True, regress the perturbation delta across cell types.
        continuous
            If True, predict across a continuous dose range.
        low_dose
            If True, predict from a treated low-dose state upwards.
        doses
            List of dose values to predict when continuous=True.

        Returns
        -------
        predicted_adata : AnnData (or dict of AnnData if continuous=True)
        delta : np.ndarray
        reg : LinearRegression (only if regression=True)
        """
        if not self.is_trained_:
            raise RuntimeError(
                "Model is not trained yet. Please call model.train() first."
            )

        # Retrieve the obs key names from the AnnDataManager registry
        batch_field = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)
        cell_type_key = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)[
            "original_key"
        ]
        treatment_key = batch_field["original_key"]

        ctrl_x = self.adata[self.adata.obs[treatment_key] == ctrl_key]
        treat_x = self.adata[self.adata.obs[treatment_key] == treat_key]
        ctrl_x = random_sample(ctrl_x, cell_type_key)
        treat_x = random_sample(treat_x, cell_type_key)

        # Balance across treatment groups
        new_adata = ctrl_x.concatenate(treat_x)
        new_adata = random_sample(
            new_adata, treatment_key, max_or_min="min", replacement=False
        )

        # Densify sparse matrix
        if sparse.issparse(new_adata.X):
            new_adata.X = new_adata.X.A

        # --- Control latent representation ---
        if not low_dose:
            ctrl_data = new_adata[
                (new_adata.obs[cell_type_key] == cell_type_to_predict)
                & (new_adata.obs[treatment_key] == ctrl_key)
            ]
        else:
            ctrl_data = new_adata[
                (new_adata.obs[cell_type_key] == cell_type_to_predict)
                & (new_adata.obs[treatment_key] == treat_key)
            ]
        latent_cd = self.get_latent_representation(ctrl_data)

        # --- Delta computation ---
        if not regression:
            ctrl_subset = new_adata[new_adata.obs[treatment_key] == ctrl_key].copy()
            treat_subset = new_adata[new_adata.obs[treatment_key] == treat_key].copy()
            latent_ctrl = np.average(
                self.get_latent_representation(ctrl_subset), axis=0
            )
            latent_treat = np.average(
                self.get_latent_representation(treat_subset), axis=0
            )
            delta = latent_treat - latent_ctrl
        else:
            latent_X = self.get_latent_representation(new_adata)
            latent_adata = sc.AnnData(X=latent_X, obs=new_adata.obs.copy())
            deltas = []
            latent_centroids = []
            cell_types = np.unique(latent_adata.obs[cell_type_key])
            for cell in cell_types:
                if cell != cell_type_to_predict:
                    lc = latent_adata[
                        (latent_adata.obs[cell_type_key] == cell)
                        & (latent_adata.obs[treatment_key] == ctrl_key)
                    ]
                    lt = latent_adata[
                        (latent_adata.obs[cell_type_key] == cell)
                        & (latent_adata.obs[treatment_key] == treat_key)
                    ]
                    ctrl_centroid = np.average(lc.X, axis=0)
                    deltas.append(np.average(lt.X, axis=0) - ctrl_centroid)
                    latent_centroids.append(ctrl_centroid)
            lr = LinearRegression()
            reg = lr.fit(latent_centroids, deltas)
            delta = reg.predict([np.average(latent_cd, axis=0)])[0]

        # --- Prediction ---
        if not continuous:
            treat_pred = delta + latent_cd
            predicted_cells = (
                self.module.generative(torch.Tensor(treat_pred).to(self.device))["px"]
                .cpu()
                .detach()
                .numpy()
            )
            predicted_adata = sc.AnnData(
                X=predicted_cells,
                obs=ctrl_data.obs.copy(),
                var=ctrl_data.var.copy(),
                obsm=ctrl_data.obsm.copy(),
            )
            if not regression:
                return predicted_adata, delta
            else:
                return predicted_adata, delta, reg

        else:
            # Continuous dose prediction
            if not low_dose:
                treat_pred_dict = {
                    d: delta * (np.log1p(d) / np.log1p(max(doses))) + latent_cd
                    for d in doses
                    if d > min(doses)
                }
            else:
                treat_pred_dict = {
                    d: latent_cd
                    - delta
                    * ((np.log1p(max(doses)) - np.log1p(d)) / np.log1p(max(doses)))
                    for d in doses
                    if d < max(doses)
                }

            dose_filter = (
                (lambda d: d > min(doses))
                if not low_dose
                else (lambda d: d < max(doses))
            )

            predicted_cells_dict = {
                d: self.module.generative(
                    torch.Tensor(treat_pred_dict[d]).to(self.device)
                )["px"]
                .cpu()
                .detach()
                .numpy()
                for d in doses
                if dose_filter(d)
            }
            predicted_adata_dict = {
                d: sc.AnnData(
                    X=predicted_cells_dict[d],
                    obs=ctrl_data.obs.copy(),
                    var=ctrl_data.var.copy(),
                    obsm=ctrl_data.obsm.copy(),
                )
                for d in doses
                if dose_filter(d)
            }

            if not regression:
                return predicted_adata_dict, delta
            else:
                return predicted_adata_dict, delta, reg

    # ------------------------------------------------------------------
    # Evaluation Plotting
    # ------------------------------------------------------------------

    def reg_mean_plot(
        self,
        adata: AnnData,
        axis_keys: dict,
        labels: dict,
        path_to_save: str = "./reg_mean.pdf",
        save: bool = True,
        gene_list: Optional[List[str]] = None,
        show: bool = False,
        top_100_genes=None,
        verbose: bool = False,
        legend: bool = True,
        title: Optional[str] = None,
        x_coeff: float = 0.30,
        y_coeff: float = 0.8,
        fontsize: int = 14,
        **kwargs,
    ):
        """
        Plot mean expression of predicted vs. ground-truth treated cells.

        Parameters
        ----------
        adata
            AnnData containing control, predicted, and (optionally) treated cells.
        axis_keys
            Dict mapping axis roles to obs condition values.
            e.g. {"x": "control", "y": "pred", "y1": "treated"}
        labels
            Dict of axis label strings. e.g. {"x": "Control", "y": "Predicted"}
        path_to_save
            File path to save the figure.
        save
            Whether to save the figure.
        gene_list
            List of gene names to annotate on the plot.
        show
            Whether to display the plot interactively.
        top_100_genes
            Optional list of top DEGs to compute a secondary R² for.
        verbose
            If True, print R² values.
        legend
            Whether to show the legend.
        title
            Plot title.
        x_coeff, y_coeff
            Position coefficients for the R² text annotation.
        fontsize
            Font size for axis labels and annotations.

        Returns
        -------
        float or tuple of floats
            R² for all genes, and optionally R² for top DEGs.
        """
        import seaborn as sns

        sns.set()
        sns.set(color_codes=True)

        if sparse.issparse(adata.X):
            adata.X = adata.X.A

        # Get the treatment obs key from registry
        condition_key = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)[
            "original_key"
        ]

        diff_genes = top_100_genes
        treat = adata[adata.obs[condition_key] == axis_keys["y"]]
        ctrl = adata[adata.obs[condition_key] == axis_keys["x"]]

        if diff_genes is not None:
            if hasattr(diff_genes, "tolist"):
                diff_genes = diff_genes.tolist()
            adata_diff = adata[:, diff_genes]
            treat_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]]
            ctrl_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]]
            x_diff = numpy.average(ctrl_diff.X, axis=0)
            y_diff = numpy.average(treat_diff.X, axis=0)
            m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(
                x_diff, y_diff
            )
            if verbose:
                print("top_100 DEGs mean: ", r_value_diff**2)

        x = numpy.average(ctrl.X, axis=0)
        y = numpy.average(treat.X, axis=0)
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        if verbose:
            print("All genes mean: ", r_value**2)

        df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df)
        ax.tick_params(labelsize=fontsize)

        if "range" in kwargs:
            start, stop, step = kwargs.get("range")
            ax.set_xticks(numpy.arange(start, stop, step))
            ax.set_yticks(numpy.arange(start, stop, step))

        ax.set_xlabel(labels["x"], fontsize=fontsize)
        ax.set_ylabel(labels["y"], fontsize=fontsize)

        if gene_list is not None:
            texts = []
            for i in gene_list:
                j = adata.var_names.tolist().index(i)
                x_bar = x[j]
                y_bar = y[j]
                texts.append(pyplot.text(x_bar, y_bar, i, fontsize=11, color="black"))
                pyplot.plot(x_bar, y_bar, "o", color="red", markersize=5)
            adjust_text(
                texts,
                x=x,
                y=y,
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.5),
                force_points=(0.0, 0.0),
            )

        if legend:
            pyplot.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        pyplot.title(title if title is not None else "", fontsize=fontsize)

        ax.text(
            max(x) - max(x) * x_coeff,
            max(y) - y_coeff * max(y),
            r"$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= " + f"{r_value**2:.2f}",
            fontsize=kwargs.get("textsize", fontsize),
        )
        if diff_genes is not None:
            ax.text(
                max(x) - max(x) * x_coeff,
                max(y) - (y_coeff + 0.15) * max(y),
                r"$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= "
                + f"{r_value_diff**2:.2f}",
                fontsize=kwargs.get("textsize", fontsize),
            )

        if save:
            pyplot.savefig(path_to_save, bbox_inches="tight", dpi=100)
        if show:
            pyplot.show()
        pyplot.close()

        if diff_genes is not None:
            return r_value**2, r_value_diff**2
        else:
            return r_value**2
