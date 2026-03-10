import scanpy as sc
import numpy as np
import pandas as pd
from collections import Counter
from scipy import sparse
from scipy import stats


def create_cell_dose_column(adata, celltype_column, dose_column):
    """Adds a column to 'adata.obs' combining cell type and condition labels.

    Concatenates values from `celltype_column` and `dose_column` with an
    underscore separator, creating unique identifiers for each combination.

    Args:
        adata (AnnData): AnnData object with metadata in `adata.obs`.
        celltype_column (str): Column name for cell type (e.g. 'celltype').
        dose_column (str): Column name for condition/dose (e.g. 'dose').

    Returns:
        pandas.Series: Concatenated values, suitable for adding as a new column.

    Example:
        >>> adata.obs['cell_condition'] = create_cell_dose_column(adata, 'celltype', 'condition')
        # 'T-cell' + 'Disease' -> 'T-cell_Disease'
    """
    return adata.obs.apply(lambda x: f"{x[celltype_column]}_{x[dose_column]}", axis=1)


def normalize_data(adata):
    """Normalizes, log-transforms, and selects 5000 highly variable genes.

    Applies scanpy's normalize_total, log1p, and highly_variable_genes
    to streamline preprocessing.

    Args:
        adata (AnnData): AnnData object with raw counts.

    Returns:
        AnnData: New AnnData object containing only the top 5000 highly variable genes,
            normalized and log-transformed.

    Example:
        >>> adata_filtered = normalize_data(adata)
    """
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=5000)
    return adata[:, adata.var.highly_variable]


def prepare_data(
    adata,
    cell_type_key,
    treatment_key,
    cell_type_to_predict,
    treatment_to_predict,
    normalized=False,
):
    """Prepares training and testing data for single-condition analysis.

    Splits an AnnData object into train and test sets. The test set contains
    cells of `cell_type_to_predict` under `treatment_to_predict`. All other
    cells form the training set.

    Note: AnnData registration (setup_anndata) is handled externally by the
    VIDR model class — do not call it here.

    Args:
        adata (AnnData): AnnData object with cell type and treatment metadata.
        cell_type_key (str): Column in `adata.obs` for cell type labels.
        treatment_key (str): Column in `adata.obs` for condition/treatment labels.
        cell_type_to_predict (str): Cell type to hold out for testing.
        treatment_to_predict (str): Treatment/condition to hold out for testing.
        normalized (bool, optional): Whether data is already normalized. Defaults to False.

    Returns:
        tuple:
            - train_adata (AnnData): All cells except the held-out cell type + condition.
            - test_adata (AnnData): Only the held-out cell type under held-out condition.

    Example:
        >>> train_adata, test_adata = prepare_data(
        ...     adata,
        ...     cell_type_key='cell_type',
        ...     treatment_key='Condition',
        ...     cell_type_to_predict='LSEC',
        ...     treatment_to_predict='Disease',
        ... )
    """
    if not normalized:
        adata = normalize_data(adata)

    held_out_mask = (adata.obs[cell_type_key] == cell_type_to_predict) & (
        adata.obs[treatment_key] == treatment_to_predict
    )
    train_adata = adata[~held_out_mask].copy()
    test_adata = adata[held_out_mask].copy()

    return train_adata, test_adata


def prepare_cont_data(
    adata,
    cell_type_key,
    treatment_key,
    dose_key,
    cell_type_to_predict,
    control_dose,
    normalized=False,
):
    """Prepares training and testing data for multi-condition (numeric) analysis.

    Splits an AnnData object into train and test sets. The test set contains
    cells of `cell_type_to_predict` with dose values greater than `control_dose`.
    All other cells form the training set.

    Note: AnnData registration (setup_anndata) is handled externally by the
    VIDR model class — do not call it here.

    Args:
        adata (AnnData): AnnData object with cell type and dose metadata.
        cell_type_key (str): Column in `adata.obs` for cell type labels.
        treatment_key (str): Column in `adata.obs` for condition labels (string version).
        dose_key (str): Column in `adata.obs` for numeric dose values.
        cell_type_to_predict (str): Cell type to hold out for testing.
        control_dose (float): Cells with dose > control_dose are held out as test set.
        normalized (bool, optional): Whether data is already normalized. Defaults to False.

    Returns:
        tuple:
            - train_adata (AnnData): All cells except held-out cell type at non-control doses.
            - test_adata (AnnData): Only held-out cell type at doses above control_dose.

    Example:
        >>> train_adata, test_adata = prepare_cont_data(
        ...     adata,
        ...     cell_type_key='cell_type',
        ...     treatment_key='condition',
        ...     dose_key='Dose',
        ...     cell_type_to_predict='Hepatocyte',
        ...     control_dose=0.0,
        ... )
    """
    if not normalized:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=5000)
        adata = adata[:, adata.var.highly_variable]

    held_out_mask = (adata.obs[cell_type_key] == cell_type_to_predict) & (
        adata.obs[dose_key] > control_dose
    )
    train_adata = adata[~held_out_mask].copy()
    test_adata = adata[held_out_mask].copy()

    return train_adata, test_adata


def calculate_r2_singledose(
    adata,
    cell: str,
    model: str,
    condition_key: str,
    axis_keys: dict,
    diff_genes=None,
    random_sample_coef=None,
    n_iter: int = 1,
):
    """Calculate R² values comparing predicted and real treated cells (single condition).

    Args:
        adata (AnnData): AnnData containing both predicted and real treated cells.
        cell (str): Cell type label for annotation in results.
        model (str): Model name for annotation in results.
        condition_key (str): Column in `adata.obs` distinguishing conditions.
        axis_keys (dict): Keys "x" (predicted) and "y" (real treated) mapping to
            condition values. e.g. {"x": "predicted_Disease", "y": "Disease"}
        diff_genes (list, optional): List of DEG names for a secondary R² calculation.
        random_sample_coef (float, optional): Fraction of cells to subsample each iteration.
        n_iter (int, optional): Number of subsampling iterations. Defaults to 1.

    Returns:
        pd.DataFrame: R² values with columns: R^2, Gene Set, Cell, Model.
    """
    if sparse.issparse(adata.X):
        adata.X = adata.X.A

    treat = adata[adata.obs[condition_key] == axis_keys["y"]]
    pred = adata[adata.obs[condition_key] == axis_keys["x"]]

    r2_values_dict = {"R^2": [], "Gene Set": []}
    for _ in range(n_iter):
        if random_sample_coef is not None:
            treat_idx = np.random.choice(
                treat.shape[0], int(random_sample_coef * treat.shape[0])
            )
            pred_idx = np.random.choice(
                pred.shape[0], int(random_sample_coef * pred.shape[0])
            )
            treat_samp = treat[treat_idx, :]
            pred_samp = pred[pred_idx, :]
        else:
            treat_samp = treat
            pred_samp = pred  # Bug fix: was undefined 'samp'

        if diff_genes is not None:
            x_diff = np.average(pred_samp[:, diff_genes].X, axis=0)
            y_diff = np.average(treat_samp[:, diff_genes].X, axis=0)
            _, _, r_diff, _, _ = stats.linregress(x_diff, y_diff)
            r2_values_dict["R^2"].append(r_diff**2)
            r2_values_dict["Gene Set"].append("DEGs")

        x = np.average(pred_samp.X, axis=0)
        y = np.average(treat_samp.X, axis=0)
        _, _, r_value, _, _ = stats.linregress(x, y)
        r2_values_dict["R^2"].append(r_value**2)
        r2_values_dict["Gene Set"].append("All HVGs")

    r2_df = pd.DataFrame(r2_values_dict)
    r2_df["Cell"] = cell
    r2_df["Model"] = model
    return r2_df


def calculate_r2_multidose(
    adata,
    cell: str,
    model: str,
    condition_key: str,
    axis_keys: dict,
    diff_genes=None,
    random_sample_coef=None,
    n_iter: int = 1,
):
    """Calculate R² values comparing predicted and real treated cells (multi-condition / dose response).

    Same as calculate_r2_singledose but includes the dose value in the output,
    allowing R² to be tracked across multiple dose levels.

    Args:
        adata (AnnData): AnnData containing both predicted and real treated cells.
        cell (str): Cell type label for annotation in results.
        model (str): Model name for annotation in results.
        condition_key (str): Column in `adata.obs` distinguishing conditions/doses.
        axis_keys (dict): Keys "x" (predicted) and "y" (real treated) mapping to
            condition values.
        diff_genes (list, optional): List of DEG names for a secondary R² calculation.
        random_sample_coef (float, optional): Fraction of cells to subsample each iteration.
        n_iter (int, optional): Number of subsampling iterations. Defaults to 1.

    Returns:
        pd.DataFrame: R² values with columns: R^2, Gene Set, Cell, Model, Dose.
    """
    if sparse.issparse(adata.X):
        adata.X = adata.X.A

    treat = adata[adata.obs[condition_key] == axis_keys["y"]]
    pred = adata[adata.obs[condition_key] == axis_keys["x"]]

    r2_values_dict = {"R^2": [], "Gene Set": []}
    for _ in range(n_iter):
        if random_sample_coef is not None:
            treat_idx = np.random.choice(
                treat.shape[0], int(random_sample_coef * treat.shape[0])
            )
            pred_idx = np.random.choice(
                pred.shape[0], int(random_sample_coef * pred.shape[0])
            )
            treat_samp = treat[treat_idx, :]
            pred_samp = pred[pred_idx, :]
        else:
            treat_samp = treat
            pred_samp = pred  # Bug fix: was undefined 'samp'

        if diff_genes is not None:
            x_diff = np.average(pred_samp[:, diff_genes].X, axis=0)
            y_diff = np.average(treat_samp[:, diff_genes].X, axis=0)
            _, _, r_diff, _, _ = stats.linregress(x_diff, y_diff)
            r2_values_dict["R^2"].append(r_diff**2)
            r2_values_dict["Gene Set"].append("DEGs")

        x = np.average(pred_samp.X, axis=0)
        y = np.average(treat_samp.X, axis=0)
        _, _, r_value, _, _ = stats.linregress(x, y)
        r2_values_dict["R^2"].append(r_value**2)
        r2_values_dict["Gene Set"].append("All HVGs")

    r2_df = pd.DataFrame(r2_values_dict)
    r2_df["Cell"] = cell
    r2_df["Model"] = model
    r2_df["Dose"] = axis_keys["y"]
    return r2_df


def random_sample(adata, key, max_or_min="max", replacement=True):
    """Randomly samples and balances cell populations by a grouping key.

    Resamples cells so each group defined by `key` has equal size, matching
    either the largest or smallest group.

    Args:
        adata (AnnData): AnnData object to resample.
        key (str): Column in `adata.obs` used to define groups.
        max_or_min (str, optional): Balance to the largest ("max") or smallest
            ("min") group size. Defaults to "max".
        replacement (bool, optional): Whether to sample with replacement.
            Always True when max_or_min="max". Defaults to True.

    Returns:
        AnnData: Resampled AnnData with equal-sized groups.

    Example:
        >>> balanced = random_sample(adata, key='cell_type', max_or_min='min', replacement=False)
    """
    pop_dict = Counter(adata.obs[key])
    eq = (
        np.max(list(pop_dict.values()))
        if max_or_min == "max"
        else np.min(list(pop_dict.values()))
    )
    replacement = True if max_or_min == "max" else replacement

    idxs = []
    for group in pop_dict.keys():
        group_idx = np.nonzero(np.array(adata.obs[key] == group))[0]
        resampled = group_idx[np.random.choice(len(group_idx), eq, replace=replacement)]
        idxs.append(resampled)

    return adata[np.concatenate(idxs)].copy()
