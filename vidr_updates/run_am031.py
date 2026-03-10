"""
Quick train + predict script for a single sample (AM031)
using the refactored scVIDR code in vidr_updates/.

Run from the vidr_updates/ directory:
    cd ~/scVIDR_Refactor/vidr_updates
    python run_am031.py
"""

import os
import sys

# Point to the refactored code in vidr_updates/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scanpy as sc
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from new_vidr import VIDR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ADATA_PATH = "/mnt/home/bowmand8/adata_healthy_diseased_nucseq.h5ad"
SAMPLE_ID = "AM031"
MODEL_DIR = "./model_AM031"
OUTPUT_DIR = "./predictions_AM031"

# Known from data exploration
CTRL_CONDITION = "Normal"
TREAT_CONDITION = "Disease"

# ---------------------------------------------------------------------------
# 1. Load and subset to AM031
# ---------------------------------------------------------------------------
logging.info(f"Loading data from {ADATA_PATH}")
adata = sc.read_h5ad(ADATA_PATH)

adata = adata[adata.obs["sample_id"] == SAMPLE_ID].copy()
logging.info(f"Cells in {SAMPLE_ID}: {adata.n_obs}")
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
logging.info(
    f"Unified cell_type breakdown:\n{adata.obs['cell_type'].value_counts().to_string()}"
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

# Pick most abundant shared cell type as the hold-out
shared_counts = adata.obs[adata.obs["cell_type"].isin(shared_celltypes)][
    "cell_type"
].value_counts()
TEST_CELLTYPE = shared_counts.index[0]
logging.info(f"Test cell type (held out from training): '{TEST_CELLTYPE}'")

# ---------------------------------------------------------------------------
# 5. Train / test split — hold out the test cell type
# ---------------------------------------------------------------------------
train_adata = adata[adata.obs["cell_type"] != TEST_CELLTYPE].copy()
test_adata = adata[adata.obs["cell_type"] == TEST_CELLTYPE].copy()

logging.info(f"Train cells: {train_adata.n_obs}")
logging.info(f"Test cells:  {test_adata.n_obs}")

# ---------------------------------------------------------------------------
# 6. Setup AnnData and initialize model
# ---------------------------------------------------------------------------
logging.info("Setting up AnnData for VIDR...")

VIDR.setup_anndata(
    train_adata,
    batch_key="Condition",  # "Normal" / "Disease" — used as ctrl_key/treat_key
    labels_key="cell_type",  # unified cell type column
    condition_key=None,  # categorical comparison, no numeric condition needed
)

model = VIDR(
    train_adata,
    n_hidden=256,  # smaller than default (800) — single sample ~4k cells
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
#    Ask: what would TEST_CELLTYPE look like under Disease condition?
# ---------------------------------------------------------------------------
logging.info(f"\nPredicting '{TEST_CELLTYPE}' under '{TREAT_CONDITION}' condition...")

pred_adata, delta = model.predict(
    ctrl_key=CTRL_CONDITION,
    treat_key=TREAT_CONDITION,
    cell_type_to_predict=TEST_CELLTYPE,
    regression=True,  # scVIDR mode: regress delta across cell types
)

pred_adata.obs["Condition"] = f"predicted_{TREAT_CONDITION}"
pred_adata.obs["cell_type"] = TEST_CELLTYPE
pred_adata.obs["Model"] = "scVIDR"

logging.info(f"Predicted cells: {pred_adata.n_obs}")

# ---------------------------------------------------------------------------
# 9. Save predictions
# ---------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, f"{SAMPLE_ID}_{TEST_CELLTYPE}_pred.h5ad")
pred_adata.write_h5ad(out_path)
logging.info(f"Saved predictions to: {out_path}")

# ---------------------------------------------------------------------------
# 10. Quick R² sanity check vs. real Disease cells
# ---------------------------------------------------------------------------
real_treated = test_adata[test_adata.obs["Condition"] == TREAT_CONDITION]

if real_treated.n_obs > 0:
    from scipy import stats, sparse

    X_pred = pred_adata.X.A if sparse.issparse(pred_adata.X) else pred_adata.X
    X_real = real_treated.X.A if sparse.issparse(real_treated.X) else real_treated.X

    mean_pred = X_pred.mean(axis=0)
    mean_real = X_real.mean(axis=0)
    _, _, r_value, _, _ = stats.linregress(mean_real, mean_pred)
    logging.info(f"R² (predicted vs. real Disease mean expression): {r_value**2:.4f}")
else:
    logging.info(
        f"No real Disease cells for '{TEST_CELLTYPE}' in test set — skipping R² check."
    )

logging.info("All done.")
