import os
import sys

sys.path.insert(1, "../vidr")

import argparse
import logging

logging.basicConfig(level=logging.INFO)

import scanpy as sc

from vidr import VIDR
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
        "Train a scVIDR model on a h5ad dataset. "
        "Conditions can be numeric (e.g. drug doses) or categorical (e.g. healthy vs. diseased)."
    )
)
subparsers = parser.add_subparsers(
    help="Train a single or multi condition model", dest="train_command"
)
parser_single = subparsers.add_parser(
    SINGLE_CONDITION_COMMAND,
    help="Train a model comparing two conditions (e.g. healthy vs. diseased, or 0mg vs. 30mg)",
)
parser_multi = subparsers.add_parser(
    MULTI_CONDITION_COMMAND,
    help="Train a model across multiple numeric conditions (e.g. dose response curve)",
)

for subparser in [parser_single, parser_multi]:
    subparser.add_argument(
        "h5ad_data_file", help="The data file containing the raw reads in h5ad format"
    )
    subparser.add_argument(
        "model_path", help="Path to the directory where the trained model will be saved"
    )
    subparser.add_argument(
        "--condition_column",
        help=(
            "Name of the condition column in adata.obs. "
            "For numeric conditions (doses), this should be the numeric column. "
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
        help='Cell type to hold out for testing (default "Hepatocytes - portal")',
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
            "Cell types to keep during training/testing. "
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
# Internal string version of condition — always used as batch_key in scvi
# (scvi requires categorical batch_key, so numeric conditions get stringified)
CONDITION_KEY = "condition"
TEST_CELLTYPE = script_args.test_celltype
CONTROL_CONDITION = script_args.control_condition
TREAT_CONDITION = script_args.treat_condition
MODEL_OUTPUT_DIR = script_args.model_path
CELLTYPES_OF_INTEREST = script_args.celltypes_keep
TRAIN_COMMAND = script_args.train_command

is_numeric = CONDITION_TYPE == "numeric"
is_single = TRAIN_COMMAND == SINGLE_CONDITION_COMMAND
is_multi = TRAIN_COMMAND == MULTI_CONDITION_COMMAND

# multi_condition only makes sense for numeric conditions (e.g. dose response)
if is_multi and not is_numeric:
    raise ValueError(
        "multi_condition mode is only supported for numeric conditions. "
        "Use single_condition with --condition_type categorical for categorical comparisons "
        "(e.g. healthy vs. diseased)."
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

# Always create a string version of condition for use as batch_key in scvi
adata.obs[CONDITION_KEY] = adata.obs[CONDITION_COLUMN].astype(str)
available_conditions = adata.obs[CONDITION_KEY].unique()


def check_condition(arg_name, value, available):
    if value not in available:
        raise ValueError(
            f'Condition "{arg_name}"={value} not found in available conditions: {available}'
        )


check_condition("control_condition", CONTROL_CONDITION, available_conditions)
if is_single:
    check_condition("treat_condition", TREAT_CONDITION, available_conditions)

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
        CONDITION_COLUMN,  # original numeric column for continuous NCA loss
        TEST_CELLTYPE,
        float(CONTROL_CONDITION),
        normalized=True,
    )

# ---------------------------------------------------------------------------
# Setup AnnData and train model
# ---------------------------------------------------------------------------
logging.info("\nSetting up AnnData and initializing VIDR model...")

# batch_key      = CONDITION_KEY       (string condition, used for ctrl/treat split in predict())
# labels_key     = CELLTYPE_COLUMN     (cell type, used for NCA loss grouping)
# condition_key  = CONDITION_COLUMN    (numeric only — for continuous NCA loss in multi mode)
VIDR.setup_anndata(
    train_adata,
    batch_key=CONDITION_KEY,
    labels_key=CELLTYPE_COLUMN,
    condition_key=CONDITION_COLUMN if is_multi else None,
)

model = VIDR(
    train_adata,
    use_linear_decoder=False,
    use_nca_loss=True,
    use_condition_loss=is_multi,  # only meaningful for numeric multi-condition
)

logging.info("\nTraining scVIDR model...")
model.train(
    max_epochs=100,
    batch_size=128,
    early_stopping=True,
    plan_kwargs={"early_stopping_patience": 25},
)
logging.info("Training done.")

logging.info(f"Saving model to: {MODEL_OUTPUT_DIR}")
model.save(MODEL_OUTPUT_DIR, overwrite=True)
logging.info("Save done.")
