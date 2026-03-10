"""
Train + predict + plot script for the liver fibrosis dataset.
Uses all samples from adata_healthy_diseased_nucseq.h5ad.

Dataset structure:
  Normal  samples: AM042, AM048, AM061
  Disease samples: AM031, AM062, AM072
  (No single sample has both conditions — must use the full dataset.)

Run from the vidr_updates/ directory:
    cd ~/scVIDR_Refactor/vidr_updates
    python run_liver_fibrosis.py
"""

import logging
import os
import sys

# Point to the refactored code in vidr_updates/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anndata
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from scipy import sparse

from new_vidr import VIDR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ADATA_PATH = "/mnt/home/bowmand8/adata_healthy_diseased_nucseq.h5ad"
MODEL_DIR = "./model_full"
OUTPUT_DIR = "./predictions_full"
FIGURES_DIR = "./liver_fibrosis_figures"

CTRL_CONDITION = "Normal"
TREAT_CONDITION = "Disease"

# ---------------------------------------------------------------------------
# 1. Load full dataset (all samples — Normal and Disease are in different samples)
# ---------------------------------------------------------------------------
logging.info(f"Loading data from {ADATA_PATH}")
adata = sc.read_h5ad(ADATA_PATH)
logging.info(f"Total cells: {adata.n_obs}")
logging.info(
    f"Condition breakdown:\n{adata.obs['Condition'].value_counts().to_string()}"
)

# ---------------------------------------------------------------------------
# 2. Build a unified cell_type column
#
# The dataset uses two separate columns for cell type depending on condition:
#   - Normal  cells → cell_type_final_healthy
#   - Disease cells → cell_type_final_injured
# We merge these into a single 'cell_type' column.
# ---------------------------------------------------------------------------
adata.obs["cell_type"] = np.where(
    adata.obs["Condition"] == CTRL_CONDITION,
    adata.obs["cell_type_final_healthy"].astype(str),
    adata.obs["cell_type_final_injured"].astype(str),
)

# ---------------------------------------------------------------------------
# 2b. Harmonise cell type labels across conditions
#
# Healthy and diseased samples use different sub-groupings for the same
# broad cell populations. Remap to a shared vocabulary so the model can
# compute cross-condition deltas.
#
#   Macrophages       ← Mac, Mac_1, Mac_2
#   HSCs              ← HSC, HSC_1, HSC_2
#   Injured_Hepatocytes ← IJ_1_Hep, IJ_2_Hep   (disease-specific state)
#   Healthy_Hepatocytes ← Hep_1, Hep_2, Hep_3, Central_Hep, Portal_Hep
#   LSEC, Cholangiocyte, Immune Cells — unchanged
# ---------------------------------------------------------------------------
CELLTYPE_REMAP = {
    # Macrophages
    "Mac":   "Macrophages",
    "Mac_1": "Macrophages",
    "Mac_2": "Macrophages",
    # HSCs
    "HSC":   "HSCs",
    "HSC_1": "HSCs",
    "HSC_2": "HSCs",
    # Injured hepatocytes (disease-specific)
    "IJ_1_Hep": "Injured_Hepatocytes",
    "IJ_2_Hep": "Injured_Hepatocytes",
    # All other hepatocyte sub-types → healthy/zonation groups
    "Hep_1":      "Healthy_Hepatocytes",
    "Hep_2":      "Healthy_Hepatocytes",
    "Hep_3":      "Healthy_Hepatocytes",
    "Central_Hep": "Healthy_Hepatocytes",
    "Portal_Hep":  "Healthy_Hepatocytes",
}
adata.obs["cell_type"] = adata.obs["cell_type"].map(
    lambda x: CELLTYPE_REMAP.get(x, x)
)
logging.info(
    f"Unified cell_type breakdown (after remapping):\n{adata.obs['cell_type'].value_counts().to_string()}"
)

# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------
logging.info("Preprocessing...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
logging.info(f"Genes after HVG selection: {adata.n_vars}")

# ---------------------------------------------------------------------------
# 4. Pick a test cell type — most abundant type shared across both conditions
# ---------------------------------------------------------------------------
ctrl_celltypes = set(
    adata.obs.loc[adata.obs["Condition"] == CTRL_CONDITION, "cell_type"].unique()
)
treat_celltypes = set(
    adata.obs.loc[adata.obs["Condition"] == TREAT_CONDITION, "cell_type"].unique()
)
shared_celltypes = ctrl_celltypes & treat_celltypes

logging.info(f"Cell types in {CTRL_CONDITION}:  {sorted(ctrl_celltypes)}")
logging.info(f"Cell types in {TREAT_CONDITION}: {sorted(treat_celltypes)}")
logging.info(f"Shared cell types: {sorted(shared_celltypes)}")

if not shared_celltypes:
    raise ValueError(
        "No shared cell types between conditions. "
        "The healthy and injured cell type labels may need to be manually mapped "
        "to a common vocabulary before running this script."
    )

# Injured_Hepatocytes only exist in Disease — they are the prediction target.
# We hold them out of training, then ask: does applying the disease delta to
# Healthy_Hepatocytes (Normal) reproduce what Injured_Hepatocytes actually look like?
TEST_CELLTYPE = "Injured_Hepatocytes"   # held out from training (ground truth)
CTRL_CELLTYPE = "Healthy_Hepatocytes"   # starting point for latent arithmetic
logging.info(f"Held-out test cell type (ground truth): '{TEST_CELLTYPE}'")
logging.info(f"Control cell type used as prediction starting point: '{CTRL_CELLTYPE}'")

# ---------------------------------------------------------------------------
# 5. Train / test split
#    Hold out ALL Injured_Hepatocytes (Disease only) — they are the ground truth.
#    Healthy_Hepatocytes stay in training so the model learns a hepatocyte
#    representation in the Normal condition.
# ---------------------------------------------------------------------------
held_out_mask = adata.obs["cell_type"] == TEST_CELLTYPE
train_adata = adata[~held_out_mask].copy()
test_adata = adata[held_out_mask].copy()

logging.info(f"Train cells: {train_adata.n_obs}")
logging.info(f"Test cells (held-out '{TEST_CELLTYPE}'): {test_adata.n_obs}")

# ---------------------------------------------------------------------------
# 6. Setup AnnData and initialize model
# ---------------------------------------------------------------------------
logging.info("Setting up AnnData for VIDR...")

VIDR.setup_anndata(
    train_adata,
    batch_key="Condition",  # "Normal" / "Disease" — used as ctrl_key/treat_key
    labels_key="cell_type",  # unified cell type column
    condition_key=None,      # categorical comparison, no numeric condition needed
)

model = VIDR(
    train_adata,
    n_hidden=256,
    n_latent=20,
    n_layers=2,
    use_linear_decoder=False,
    use_nca_loss=True,
    use_condition_loss=False,
)
logging.info(f"\n{model}")

# ---------------------------------------------------------------------------
# 7. Train
# ---------------------------------------------------------------------------
logging.info("Training...")
model.train(
    max_epochs=100,
    batch_size=128,
    early_stopping=True,
    early_stopping_patience=15,
)

os.makedirs(MODEL_DIR, exist_ok=True)
model.save(MODEL_DIR, overwrite=True)
logging.info(f"Model saved to: {MODEL_DIR}")

# ---------------------------------------------------------------------------
# 8. Predict
#    Apply the disease delta to Healthy_Hepatocytes (Normal) and compare
#    the result against the held-out Injured_Hepatocytes (Disease).
# ---------------------------------------------------------------------------
logging.info(
    f"\nPredicting '{CTRL_CELLTYPE}' → Disease using learned delta, "
    f"comparing against held-out '{TEST_CELLTYPE}'..."
)

pred_adata, delta, reg = model.predict(
    ctrl_key=CTRL_CONDITION,
    treat_key=TREAT_CONDITION,
    cell_type_to_predict=CTRL_CELLTYPE,  # start from Healthy_Hepatocytes
    regression=True,                      # scVIDR mode: regress delta across cell types
)

pred_adata.obs["Condition"] = f"predicted_{TREAT_CONDITION}"
pred_adata.obs["cell_type"] = f"predicted_{TEST_CELLTYPE}"
pred_adata.obs["Model"] = "scVIDR"

logging.info(f"Predicted cells: {pred_adata.n_obs}")

# ---------------------------------------------------------------------------
# 9. Save predictions
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, f"{TEST_CELLTYPE}_pred.h5ad")
pred_adata.write_h5ad(out_path)
logging.info(f"Saved predictions to: {out_path}")

# ---------------------------------------------------------------------------
# 10. R² check — predicted Injured_Hepatocytes vs. real Injured_Hepatocytes
# ---------------------------------------------------------------------------
if test_adata.n_obs > 0:
    from scipy import stats, sparse

    X_pred = pred_adata.X.A if sparse.issparse(pred_adata.X) else pred_adata.X
    X_real = test_adata.X.A if sparse.issparse(test_adata.X) else test_adata.X

    mean_pred = X_pred.mean(axis=0)
    mean_real = X_real.mean(axis=0)
    _, _, r_value, _, _ = stats.linregress(mean_real, mean_pred)
    logging.info(
        f"R² (predicted {TEST_CELLTYPE} vs. real {TEST_CELLTYPE}): {r_value**2:.4f}"
    )
else:
    logging.info(f"No held-out '{TEST_CELLTYPE}' cells — skipping R² check.")

# ---------------------------------------------------------------------------
# 11. Plotting
# ---------------------------------------------------------------------------
from scipy import sparse

logging.info("Generating figures...")
os.makedirs(FIGURES_DIR, exist_ok=True)

# --- Build combined AnnData of hepatocytes for expression-space plots ---
# healthy_hep : Healthy_Hepatocytes, Condition = "Normal"
# test_adata  : Injured_Hepatocytes,           Condition = "Disease"
# pred_adata  : predicted_Injured_Hepatocytes, Condition = "predicted_Disease"
healthy_hep = train_adata[train_adata.obs["cell_type"] == CTRL_CELLTYPE].copy()

for ad in [healthy_hep, test_adata, pred_adata]:
    if sparse.issparse(ad.X):
        ad.X = ad.X.toarray()

hep_adata = anndata.concat(
    [healthy_hep, test_adata, pred_adata],
    join="inner",
)
hep_adata.obs_names_make_unique()

# --- Compute top DEGs between Normal and real Disease hepatocytes ---
deg_adata = anndata.concat([healthy_hep, test_adata], join="inner")
deg_adata.obs_names_make_unique()
sc.tl.rank_genes_groups(
    deg_adata, groupby="Condition", groups=["Disease"], reference="Normal", method="t-test"
)
top_degs = deg_adata.uns["rank_genes_groups"]["names"]["Disease"][:100].tolist()
# Keep only genes present in hep_adata after the inner join with pred_adata
top_degs = [g for g in top_degs if g in hep_adata.var_names]
logging.info(f"Top DEGs retained after gene intersection: {len(top_degs)}")
logging.info(f"Top 5 DEGs (Disease vs Normal hepatocytes): {top_degs[:5]}")

# --- Plot 1: reg_mean_plot — Normal vs Predicted ---
# Shows the magnitude and direction of the learned disease shift.
logging.info("Saving: reg_mean_normal_vs_predicted.pdf")
model.reg_mean_plot(
    hep_adata,
    axis_keys={"x": "Normal", "y": "predicted_Disease"},
    labels={"x": "Healthy Hepatocytes (Normal)", "y": "Predicted Injured Hepatocytes"},
    title="Normal vs Predicted Disease Hepatocytes",
    top_100_genes=top_degs,
    path_to_save=os.path.join(FIGURES_DIR, "reg_mean_normal_vs_predicted.pdf"),
    save=True,
    show=False,
    verbose=True,
)

# --- Plot 2: reg_mean_plot — Real Disease vs Predicted ---
# Shows prediction accuracy against held-out ground truth.
logging.info("Saving: reg_mean_real_vs_predicted.pdf")
model.reg_mean_plot(
    hep_adata,
    axis_keys={"x": "Disease", "y": "predicted_Disease"},
    labels={"x": "Real Injured Hepatocytes", "y": "Predicted Injured Hepatocytes"},
    title="Real vs Predicted Injured Hepatocytes",
    top_100_genes=top_degs,
    path_to_save=os.path.join(FIGURES_DIR, "reg_mean_real_vs_predicted.pdf"),
    save=True,
    show=False,
    verbose=True,
)

# --- Plot 3: UMAP of hepatocytes in expression space ---
# Shows how Normal, Predicted, and Real Disease hepatocytes cluster.
logging.info("Saving: umap_hepatocytes_condition.pdf  /  umap_hepatocytes_celltype.pdf")
sc.pp.pca(hep_adata)
sc.pp.neighbors(hep_adata)
sc.tl.umap(hep_adata)

sc.pl.umap(hep_adata, color="Condition", title="Hepatocytes by Condition", show=False)
plt.savefig(
    os.path.join(FIGURES_DIR, "umap_hepatocytes_condition.pdf"), bbox_inches="tight", dpi=100
)
plt.close()

sc.pl.umap(hep_adata, color="cell_type", title="Hepatocytes by Cell Type", show=False)
plt.savefig(
    os.path.join(FIGURES_DIR, "umap_hepatocytes_celltype.pdf"), bbox_inches="tight", dpi=100
)
plt.close()

# --- Plot 4: UMAP of full latent space (all training cell types + held-out test cells) ---
# Shows the NCA-structured latent organisation across all cell types.
logging.info("Saving: umap_latent_condition.pdf  /  umap_latent_celltype.pdf")

latent_train = model.get_latent_representation(train_adata)

# test_adata contains Injured_Hepatocytes — a label unseen during training.
# scvi's registry rejects unknown categories, so we encode directly via the
# module rather than going through _validate_anndata.
import torch

model.module.eval()
X_test = test_adata.X.toarray() if sparse.issparse(test_adata.X) else test_adata.X
with torch.no_grad():
    z_test = model.module.inference(
        torch.FloatTensor(X_test).to(model.device)
    )["z"].cpu().numpy()

latent_all = np.concatenate([latent_train, z_test], axis=0)
obs_all = anndata.concat(
    [train_adata, test_adata], join="inner"
).obs[["Condition", "cell_type"]].copy()
obs_all.index = [str(i) for i in range(len(obs_all))]

latent_adata = sc.AnnData(X=latent_all, obs=obs_all)
sc.pp.neighbors(latent_adata, use_rep="X")
sc.tl.umap(latent_adata)

sc.pl.umap(latent_adata, color="Condition", title="Latent Space by Condition", show=False)
plt.savefig(
    os.path.join(FIGURES_DIR, "umap_latent_condition.pdf"), bbox_inches="tight", dpi=100
)
plt.close()

sc.pl.umap(latent_adata, color="cell_type", title="Latent Space by Cell Type", show=False)
plt.savefig(
    os.path.join(FIGURES_DIR, "umap_latent_celltype.pdf"), bbox_inches="tight", dpi=100
)
plt.close()

logging.info(f"All figures saved to: {FIGURES_DIR}")
logging.info("All done.")
