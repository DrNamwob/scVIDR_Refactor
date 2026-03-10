import os
import sys

sys.path.insert(1, "../vidr")

import argparse
import logging

logging.basicConfig(level=logging.INFO)

import scanpy as sc

from new_vidr import VIDR
from new_utils import normalize_data, prepare_data, prepare_cont_data

# requirements
# input: h5ad ann datafile (scRNAseq counts)
#     - with CONDITION_COLUMN representing the condition to compare
#       (e.g. numeric dose values, or categorical labels like "healthy"/"diseased")
#     - with CELLTYPE_COLUMN representing cell type

SINGLE_CONDITION_COMMAND = "single_condition"
MULTI_CONDITION_COMMAND = "multi_condition"

parser = argparse.ArgumentParser(
    description=(
        "Create cell predictions using a pretrained scVIDR model. "
        "Conditions can be numeric (e.g. drug doses) or categorical (e.g. healthy vs. diseased)."
    )
)
subparsers = parser.add_subparsers(
    help="Predict using a single or multi condition model", dest="train_command"
)
parser_single = subparsers.add_parser(
    SINGLE_CONDITION_COMMAND,
    help="Predict using a model comparing two conditions (e.g. healthy vs. diseased)",
)
parser_multi = subparsers.add_parser(
    MULTI_CONDITION_COMMAND,
    help="Predict using a model across multiple numeric conditions (e.g. dose response curve)",
)

for subparser in [parser_single, parser_multi]:
    subparser.add_argument(
        "h5ad_data_file", help="The data file containing the raw reads in h5ad format"
    )
    subparser.add_argument(
        "model_path", help="Path to the directory where the trained model was saved"
    )
    subparser.add_argument(
        "output_path",
        help="Path to the directory where predictions will be written as h5ad files",
    )
    subparser.add_argument(
        "--model",
        help='Use scVIDR (regression delta) or scGen (mean delta) for prediction (default "scVIDR")',
        default="scVIDR",
    )
    subparser.add_argument(
        "--condition_column",
        help=(
            "Name of the condition column in adata.obs. "
            "For numeric conditions, this is the numeric column. "
            'For categorical conditions, this is the label column (e.g. "healthy"/"diseased"). '
            '(default "Condition")'
        ),
        default="Condition",
    )
    subparser.add_argument(
        "--condition_type",
        help='Whether the condition is numeric or categorical (default "categorical")',
        choices=["numeric", "categorical"],
        default="categorical",
    )
    subparser.add_argument(
        "--celltype_column",
        help='Name of the cell type column in adata.obs (default "celltype")',
        default="celltype",
    )
    subparser.add_argument(
        "--test_celltype",
        help='Cell type to predict (default "Hepatocytes - portal")',
        default="Hepatocytes - portal",
    )
    subparser.add_argument(
        "--control_condition", help='Control condition value (default "0")', default="0"
    )
    subparser.add_argument(
        "--treat_condition",
        help='Treat condition value for single condition mode (default "30")',
        default="30",
    )
    subparser.add_argument(
        "--celltypes_keep",
        help=(
            "Cell types to keep. "
            "Either a file with one cell type per line, or a semicolon-separated list "
            "(surround in quotes). Default: all available cell types."
        ),
        default="ALL",
    )

script_args = parser.parse_args()

# Load CLI arguments
DATA_PATH = script_args.h5ad_data_file
CELLTYPE_COLUMN = script_args.celltype_column
CONDITION_COLUMN = script_args.condition_column
CONDITION_TYPE = script_args.condition_type
CONDITION_KEY = "condition"  # internal string version, used as batch_key
TEST_CELLTYPE = script_args.test_celltype
CONTROL_CONDITION = script_args.control_condition
TREAT_CONDITION = script_args.treat_condition
MODEL_OUTPUT_DIR = script_args.model_path
H5AD_OUTPUT_DIR = script_args.output_path
CELLTYPES_OF_INTEREST = script_args.celltypes_keep
MODEL = script_args.model
TRAIN_COMMAND = script_args.train_command

is_numeric = CONDITION_TYPE == "numeric"
is_single = TRAIN_COMMAND == SINGLE_CONDITION_COMMAND
is_multi = TRAIN_COMMAND == MULTI_CONDITION_COMMAND

if is_multi and not is_numeric:
    raise ValueError(
        "multi_condition mode is only supported for numeric conditions. "
        "Use single_condition with --condition_type categorical for categorical comparisons."
    )

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
logging.info(f"Loading data file: {DATA_PATH}\n")
adata = sc.read_h5ad(DATA_PATH)

# ---------------------------------------------------------------------------
# Condition checks
# ---------------------------------------------------------------------------
if CONDITION_COLUMN == CONDITION_KEY:
    raise ValueError(
        f'Column name "{CONDITION_COLUMN}" is reserved for internal processing. '
        "Please rename your condition column."
    )

# Always create a string version of condition for use as batch_key
adata.obs[CONDITION_KEY] = adata.obs[CONDITION_COLUMN].astype(str)
available_conditions_str = adata.obs[CONDITION_KEY].unique()
available_conditions = adata.obs[CONDITION_COLUMN].unique()


def check_condition(arg_name, value, available):
    if value not in available:
        raise ValueError(
            f'Condition "{arg_name}"={value} not found in available conditions: {available}'
        )


check_condition("control_condition", CONTROL_CONDITION, available_conditions_str)
if is_single:
    check_condition("treat_condition", TREAT_CONDITION, available_conditions_str)

if is_single and len(available_conditions) < 2:
    raise ValueError(
        f"single_condition mode requires at least 2 conditions "
        f"(found {len(available_conditions)})."
    )
if is_multi and len(available_conditions) < 3:
    raise ValueError(
        f"multi_condition mode requires at least 3 conditions "
        f"(found {len(available_conditions)})."
    )

logging.info("Conditions available in the dataset:")
for cond in available_conditions:
    logging.info(f"  - {cond}")

# ---------------------------------------------------------------------------
# Cell type checks
# ---------------------------------------------------------------------------
available_cell_types = adata.obs[CELLTYPE_COLUMN].unique()

if CELLTYPES_OF_INTEREST != "ALL":
    if os.path.exists(CELLTYPES_OF_INTEREST):
        with open(CELLTYPES_OF_INTEREST) as f:
            CELLTYPES_OF_INTEREST = f.read().strip().split("\n")
    else:
        CELLTYPES_OF_INTEREST = CELLTYPES_OF_INTEREST.split(";")

logging.info("Cell types available in the dataset:")
for cell_type in available_cell_types:
    logging.info(f"  - {cell_type}")

if CELLTYPES_OF_INTEREST == "ALL":
    CELLTYPES_OF_INTEREST = available_cell_types

logging.info("Verifying cell types of interest...")
for cell_type in CELLTYPES_OF_INTEREST:
    if cell_type not in available_cell_types:
        raise ValueError(f'Unknown cell type of interest: "{cell_type}"')
    logging.info(f"  - {cell_type} OK")
logging.info("Cell type verification done.")

# ---------------------------------------------------------------------------
# Normalize and prepare data
# ---------------------------------------------------------------------------
logging.info("\nNormalizing and preparing data...")
adata = adata[adata.obs[CELLTYPE_COLUMN].isin(CELLTYPES_OF_INTEREST)]
adata = normalize_data(adata)
logging.info("Normalization done.")

if is_single:
    train_adata, test_adata = prepare_data(
        adata,
        CELLTYPE_COLUMN,
        CONDITION_KEY,
        TEST_CELLTYPE,
        TREAT_CONDITION,
        normalized=True,
    )

if is_multi:
    train_adata, test_adata = prepare_cont_data(
        adata,
        CELLTYPE_COLUMN,
        CONDITION_KEY,
        CONDITION_COLUMN,
        TEST_CELLTYPE,
        float(CONTROL_CONDITION),
        normalized=True,
    )

# ---------------------------------------------------------------------------
# Setup AnnData and load model
# ---------------------------------------------------------------------------
logging.info(f"\nLoading scVIDR model from: {MODEL_OUTPUT_DIR}")

# setup_anndata must be called before loading to register the same field
# mappings the model was trained with
VIDR.setup_anndata(
    train_adata,
    batch_key=CONDITION_KEY,
    labels_key=CELLTYPE_COLUMN,
    condition_key=CONDITION_COLUMN if is_multi else None,
)

model = VIDR.load(MODEL_OUTPUT_DIR, adata=train_adata)
logging.info("Model loaded.")

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def model_predict(
    model,
    control_condition,
    treat_condition,
    test_celltype,
    model_name,
    condition_column_type,
    conditions=None,
    multi_condition=False,
):
    """Run latent space arithmetic prediction and return a dict of {condition: AnnData}."""
    if model_name == "scGen":
        regression = False
    elif model_name == "scVIDR":
        regression = True
    else:
        raise ValueError(
            f'Unknown model name: "{model_name}". Choose "scGen" or "scVIDR".'
        )

    kwargs = {}
    if multi_condition:
        kwargs = {"continuous": True, "doses": conditions}

    pred, delta, *other = model.predict(
        ctrl_key=control_condition,
        treat_key=treat_condition,
        cell_type_to_predict=test_celltype,
        regression=regression,
        **kwargs,
    )
    reg = other[0] if other else None

    # Annotate predictions and normalise to dict format
    if multi_condition:
        for cond in pred.keys():
            pred[cond].obs[CONDITION_COLUMN] = cond
            pred[cond].obs[CONDITION_KEY] = str(cond)
            pred[cond].obs["Model"] = model_name
    else:
        pred.obs[CONDITION_COLUMN] = treat_condition
        pred.obs[CONDITION_COLUMN] = pred.obs[CONDITION_COLUMN].astype(
            condition_column_type
        )
        pred.obs[CONDITION_KEY] = str(treat_condition)
        pred.obs["Model"] = model_name
        pred = {pred.obs[CONDITION_COLUMN][0]: pred}

    return pred, delta, reg


condition_column_type = adata.obs[CONDITION_COLUMN].dtype

logging.info("\nRunning predictions...")
pred, delta, reg = model_predict(
    model=model,
    control_condition=CONTROL_CONDITION,
    treat_condition=TREAT_CONDITION,
    test_celltype=TEST_CELLTYPE,
    model_name=MODEL,
    condition_column_type=condition_column_type,
    conditions=available_conditions,
    multi_condition=is_multi,
)
logging.info("Predictions done.")

# ---------------------------------------------------------------------------
# Save predictions
# ---------------------------------------------------------------------------
os.makedirs(H5AD_OUTPUT_DIR, exist_ok=True)

for cond in pred.keys():
    out_path = os.path.join(H5AD_OUTPUT_DIR, f"{cond}_PRED.h5ad")
    pred[cond].write_h5ad(out_path)
    logging.info(f"Saved predicted output: {out_path}")

logging.info("All done.")
