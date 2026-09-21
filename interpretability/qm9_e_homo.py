# =============================================================================
# FULL QM9 PARTITION-INTERPRETABILITY WORKFLOW -- BUHITO IMPLEMENTATION
# =============================================================================

import sys
from pathlib import Path
import io
import re
import time
import contextlib
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from rdkit import Chem

from buhito.featurizers.bfs_graphlet_featurizer import BFSGraphletFeaturizer
from buhito.transformers import GraphletTransformer
from buhito.converters import smiles_to_nx
from buhito.regressors import HierarchicalResidualModel
from buhito.graphlet_dags import GraphletDAG, draw_projected_coefs

warnings.filterwarnings("ignore")

repo_root = Path.cwd().resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

print("Repository root:", repo_root)
print("Buhito imports successful")

repo_root = Path.cwd().resolve().parent

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Create folder for saved figures
images_dir = repo_root / "images"
images_dir.mkdir(parents=True, exist_ok=True)

print("Repository root:", repo_root)
print("Images folder:", images_dir)

from pathlib import Path

repo_root = Path.cwd().resolve().parent

csv_dir = repo_root / "csv"
csv_dir.mkdir(parents=True, exist_ok=True)

print("CSV folder:", csv_dir)


# =============================================================================
# BUHITO QM9 e_HOMO WORKFLOW
# Train once, save models/features/results, and reuse later
# =============================================================================

import sys
from pathlib import Path
import io
import re
import time
import contextlib
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from scipy import sparse

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from rdkit import Chem

from buhito.featurizers.bfs_graphlet_featurizer import BFSGraphletFeaturizer
from buhito.transformers import GraphletTransformer
from buhito.converters import smiles_to_nx
from buhito.regressors import HierarchicalResidualModel
from buhito.graphlet_dags import GraphletDAG, draw_projected_coefs


# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================
# Existing saved artifacts are reused automatically. Missing artifacts are
# generated and saved. Delete an artifact manually if you want to rebuild it.

# Assumes this notebook/script is located in:
#
# buhito-interpretability/
#     interpretability/
#
# Therefore, the repository root is one level above the current directory.

repo_root = Path.cwd().resolve().parent

# Output folders
images_dir = repo_root / "images"
csv_dir = repo_root / "csv"
models_dir = repo_root / "models"
features_dir = repo_root / "features"

# Create folders if they do not already exist
images_dir.mkdir(parents=True, exist_ok=True)
csv_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)
features_dir.mkdir(parents=True, exist_ok=True)

print("Repository root:", repo_root)
print("Images folder:", images_dir)
print("CSV folder:", csv_dir)
print("Models folder:", models_dir)
print("Features folder:", features_dir)


# =============================================================================
# CACHE / REUSE HELPERS
# =============================================================================
# This script is safe to rerun. Existing artifacts are loaded instead of
# regenerated. Delete an artifact if you intentionally want to rebuild it.

def load_sparse_or_dense(path):
    """Load a feature matrix saved by this workflow."""
    if sparse.issparse(path):
        return path
    return sparse.load_npz(path)


def save_feature_matrix(X, path):
    """Save sparse or dense feature matrices."""
    if sparse.issparse(X):
        sparse.save_npz(path, X)
    else:
        np.savez_compressed(path, X_train=np.asarray(X))


def load_feature_matrix(path, key="X_train"):
    """Load either a sparse .npz or dense npz feature matrix."""
    try:
        return sparse.load_npz(path)
    except Exception:
        return np.load(path)[key]


BUHITO_MAX_LEN = 5


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parity_plot_train_test(
    X_sets,
    y_sets,
    fitted_model,
    figsize=(12, 4),
    xlab="Observed",
    ylab="Predicted",
    title="Parity plot",
    bins=None,
):
    """Local replacement for minervachem.plotting.parity_plot_train_test."""

    fig, axes = plt.subplots(
        1,
        len(X_sets),
        figsize=figsize,
        squeeze=False,
    )

    axes = axes.ravel()

    split_names = (
        ["Train", "Test"]
        if len(X_sets) == 2
        else [f"Split {i+1}" for i in range(len(X_sets))]
    )

    for ax, X, y, split_name in zip(
        axes,
        X_sets,
        y_sets,
        split_names,
    ):
        pred = fitted_model.predict(X)

        lo = float(min(np.min(y), np.min(pred)))
        hi = float(max(np.max(y), np.max(pred)))

        ax.scatter(
            y,
            pred,
            s=8,
            alpha=0.35,
        )

        ax.plot(
            [lo, hi],
            [lo, hi],
            "--",
            linewidth=1,
        )

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

        ax.set_title(
            f"{split_name}: R^2={r2_score(y, pred):.4f}"
        )

        ax.grid(alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    return fig, axes


def smiles_to_buhito_graph(
    smiles,
    add_hs=True,
    output_2d_pos=False,
):
    result = smiles_to_nx(
        smiles,
        add_hs=add_hs,
        output_2d_pos=output_2d_pos,
    )

    return (
        result[0]
        if isinstance(result, (tuple, list))
        else result
    )


# =============================================================================
# 1. PREPROCESSING THE DATA
# =============================================================================

csv_path = Path(
    "../examples/qm9/qm9data/qm9_processed.csv"
)

print("QM9 CSV:", csv_path.resolve())

if not csv_path.exists():
    raise FileNotFoundError(
        f"Could not find QM9 CSV at:\n{csv_path.resolve()}"
    )

df = pd.read_csv(csv_path)

print("Data frame shape:", df.shape)
print("Data frame columns:", df.columns.tolist())

print("\nExample molecule:")
print(df.iloc[0])


start = time.time()

df["mol"] = df["smiles"].map(
    lambda s: Chem.AddHs(
        Chem.MolFromSmiles(s)
    )
)

df["graph"] = df["smiles"].map(
    lambda s: smiles_to_buhito_graph(
        s,
        add_hs=True,
        output_2d_pos=False,
    )
)

print(
    f"Time to create {len(df)} RDKit molecules "
    f"+ Buhito graphs: "
    f"{time.time() - start:.2f} seconds"
)


for idx in range(min(10, len(df))):

    mol = df.iloc[idx]["mol"]
    graph = df.iloc[idx]["graph"]

    assert set(map(int, graph.nodes)) == set(
        range(mol.GetNumAtoms())
    )


print(
    "Buhito graph node IDs match "
    "RDKit atom indices"
)


# Display first molecule in notebook
df["mol"].iloc[0]


# =============================================================================
# 2. FEATURE ENGINEERING WITH BUHITO GRAPHLET FINGERPRINTS
# =============================================================================

base_graphlet_featurizer = BFSGraphletFeaturizer(
    return_nodewise=False,
    max_len=BUHITO_MAX_LEN,
)

featurizer = GraphletTransformer(
    featurizer=base_graphlet_featurizer,
    verbose=0,
    n_jobs=-1,
    chunk_size="auto",
)

print("Buhito featurizer initialized")
print("Max graphlet length:", BUHITO_MAX_LEN)


# -------------------------------------------------------------------------
# Train/test split
# -------------------------------------------------------------------------

split_path = csv_dir / "buhito_e_homo_train_test_split.csv"

if split_path.exists():
    print("\nExisting train/test split found. Loading:")
    print(" ", split_path)

    split_df = pd.read_csv(split_path)

    train_smiles = split_df.loc[
        split_df["split"] == "train", "smiles"
    ].tolist()
    test_smiles = split_df.loc[
        split_df["split"] == "test", "smiles"
    ].tolist()

    df_by_smiles = df.set_index("smiles", drop=False)

    train = df_by_smiles.loc[train_smiles].reset_index(drop=True)
    test = df_by_smiles.loc[test_smiles].reset_index(drop=True)

else:
    print("\nNo saved train/test split found. Creating it...")

    train, test = train_test_split(
        df,
        train_size=0.8,
        random_state=42,
    )

    split_df = pd.concat(
        [
            train[["smiles"]].assign(split="train"),
            test[["smiles"]].assign(split="test"),
        ],
        ignore_index=True,
    )

    split_df.to_csv(split_path, index=False)

    print("Saved train/test split:")
    print(" ", split_path)

print(f"Training set: {len(train)} molecules")
print(f"Test set:     {len(test)} molecules")


# -------------------------------------------------------------------------
# Buhito graphlet features + fitted featurizer
# -------------------------------------------------------------------------

X_train_path = features_dir / "X_train_buhito.npz"
X_test_path = features_dir / "X_test_buhito.npz"
featurizer_path = models_dir / "buhito_featurizer.pkl"

if (
    X_train_path.exists()
    and X_test_path.exists()
    and featurizer_path.exists()
):
    print("\nExisting Buhito features and featurizer found. Loading...")

    featurizer = joblib.load(featurizer_path)
    X_train = load_feature_matrix(X_train_path, key="X_train")
    X_test = load_feature_matrix(X_test_path, key="X_test")

else:
    print("\nSaved features/featurizer not found. Generating...")

    start = time.time()

    X_train = featurizer.fit_transform(
        train["graph"].tolist()
    )

    X_test = featurizer.transform(
        test["graph"].tolist()
    )

    print(
        f"Time to generate graphlet features: "
        f"{time.time() - start:.2f} seconds"
    )

    save_feature_matrix(X_train, X_train_path)

    if sparse.issparse(X_test):
        sparse.save_npz(X_test_path, X_test)
    else:
        np.savez_compressed(
            X_test_path,
            X_test=np.asarray(X_test),
        )

    joblib.dump(featurizer, featurizer_path)

    print("Saved feature matrices:")
    print(" ", X_train_path)
    print(" ", X_test_path)

    print("Saved fitted Buhito featurizer:")
    print(" ", featurizer_path)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Fitted graphlet bits:", len(featurizer.bit_ids_))
print(
    "Max fitted bit size:",
    int(np.max(featurizer.bit_sizes_)),
)


# =============================================================================
# STABLE GRAPH LOOKUP
# =============================================================================

def _mol_lookup_key(mol):

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )


_GRAPH_BY_MOL_KEY = {
    _mol_lookup_key(mol): graph
    for mol, graph in zip(
        df["mol"],
        df["graph"],
    )
}


def graph_for_rdkit_mol(mol):

    key = _mol_lookup_key(mol)

    if key not in _GRAPH_BY_MOL_KEY:

        smiles = Chem.MolToSmiles(
            Chem.RemoveHs(mol),
            canonical=False,
        )

        graph = smiles_to_buhito_graph(
            smiles,
            add_hs=True,
            output_2d_pos=False,
        )

        if set(map(int, graph.nodes)) != set(
            range(mol.GetNumAtoms())
        ):
            raise ValueError(
                "Generated Buhito graph node IDs "
                "do not align with RDKit atom indices"
            )

        _GRAPH_BY_MOL_KEY[key] = graph

    return _GRAPH_BY_MOL_KEY[key]


def transform_rdkit_mols(mols):

    return featurizer.transform(
        [
            graph_for_rdkit_mol(mol)
            for mol in mols
        ]
    )


def make_buhito_dag(
    mol,
    graph=None,
    coefficients=None,
    layerwise=True,
):
    """Build a Buhito GraphletDAG from Buhito GraphletTransformer bit-info."""

    if graph is None:
        graph = graph_for_rdkit_mol(mol)

    # Transform one graph so bi_transform_[0]
    # corresponds to this molecule.
    featurizer.transform([graph])

    if (
        not hasattr(featurizer, "bi_transform_")
        or len(featurizer.bi_transform_) != 1
    ):
        raise RuntimeError(
            "Buhito GraphletTransformer did not expose "
            "one bi_transform_ mapping"
        )

    bi = featurizer.bi_transform_[0]

    if coefficients is None:
        coefficients = partition_model.coef_

    return GraphletDAG(
        mol=mol,
        bi=bi,
        layerwise=layerwise,
        bit_ids=list(featurizer.bit_ids_),
        coef=np.asarray(
            coefficients,
            dtype=float,
        ).reshape(-1),
    )


# =============================================================================
# 3. RIDGE REGRESSION BASELINE
# =============================================================================

y_train = train["e_homo"].values
y_test = test["e_homo"].values

print(
    "Target variable: e_homo "
    "(Highest Occupied Molecular Orbital Energy)"
)

print(
    f"  Train range: "
    f"[{y_train.min():.2f}, {y_train.max():.2f}] eV"
)

print(
    f"  Test range:  "
    f"[{y_test.min():.2f}, {y_test.max():.2f}] eV"
)


model = Ridge(
    fit_intercept=False,
    alpha=1e-5,
    solver="sparse_cg",
    tol=1e-5,
)

ridge_model_path = models_dir / "buhito_ridge_model.pkl"

if ridge_model_path.exists():
    print("\nExisting Ridge model found. Loading:")
    print(" ", ridge_model_path)
    model = joblib.load(ridge_model_path)
else:
    print("\nNo saved Ridge model found. Training...")

    start = time.time()

    model.fit(
        X_train,
        y_train,
    )

    print(
        f"Training time: "
        f"{time.time() - start:.2f} seconds"
    )

    joblib.dump(model, ridge_model_path)

    print("Saved Ridge model:")
    print(" ", ridge_model_path)


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


train_r2 = r2_score(
    y_train,
    y_train_pred,
)

test_r2 = r2_score(
    y_test,
    y_test_pred,
)

train_mae = mean_absolute_error(
    y_train,
    y_train_pred,
)

test_mae = mean_absolute_error(
    y_test,
    y_test_pred,
)


print("\nPerformance Metrics:")

print(
    f"  Train R^2:  {train_r2:.4f}"
)

print(
    f"  Test R^2:   {test_r2:.4f}"
)

print(
    f"  Train MAE: {train_mae:.2f} eV"
)

print(
    f"  Test MAE:  {test_mae:.2f} eV"
)



# -------------------------------------------------------------------------
# Ridge parity plot
# -------------------------------------------------------------------------

parity_plot_train_test(
    [X_train, X_test],
    [y_train, y_test],
    model,
    figsize=(12, 4),
    xlab="DFT e_HOMO (eV)",
    ylab="Ridge Model e_HOMO (eV)",
    title="QM9 e_HOMO Prediction - Buhito Ridge Baseline",
    bins=75,
)

plt.savefig(
    images_dir / "ridge_perf_buhito_e_homo.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()


# =============================================================================
# 4. BUHITO HIERARCHICAL RESIDUAL MODEL
# =============================================================================

base_model = Ridge(
    fit_intercept=False,
    alpha=1e-5,
    solver="sparse_cg",
)

hmodel = HierarchicalResidualModel(
    regressor=base_model,
    verbose=1,
)


hmodel_path = models_dir / "buhito_hierarchical_model.pkl"

if hmodel_path.exists():
    print("\nExisting hierarchical model found. Loading:")
    print(" ", hmodel_path)
    hmodel = joblib.load(hmodel_path)
else:
    print("\nNo saved hierarchical model found. Training...")

    start = time.time()

    hmodel.fit(
        X_train,
        y_train,
        levels=np.asarray(
            featurizer.bit_sizes_
        ),
    )

    print(
        f"Training time: "
        f"{time.time() - start:.2f} seconds"
    )

    joblib.dump(hmodel, hmodel_path)

    print("Saved hierarchical model:")
    print(" ", hmodel_path)

print(
    "X_train shape:",
    X_train.shape,
)

print(
    "hmodel.coef_ shape:",
    hmodel.coef_.shape,
)

assert len(
    np.asarray(
        hmodel.coef_
    ).reshape(-1)
) == len(
    featurizer.bit_ids_
)


# -------------------------------------------------------------------------
# Put hierarchical coefficients in feature order
# -------------------------------------------------------------------------

def hierarchical_coef_in_feature_order(
    fitted_hmodel,
    levels,
):

    levels = np.asarray(levels)

    coef = np.zeros(
        levels.shape[0],
        dtype=float,
    )

    for level in fitted_hmodel.unique_levels:

        level_coef = np.asarray(
            fitted_hmodel.regressors[level].coef_,
            dtype=float,
        ).reshape(-1)

        mask = levels == level

        if int(mask.sum()) != len(level_coef):

            raise ValueError(
                f"Level {level}: "
                f"{int(mask.sum())} feature columns but "
                f"{len(level_coef)} fitted coefficients"
            )

        coef[mask] = level_coef

    return coef


class HierarchicalModelFeatureOrderView:

    """Prediction-compatible view with coef_
    aligned to Buhito feature columns.
    """

    def __init__(
        self,
        fitted_hmodel,
        levels,
    ):

        self.fitted_hmodel = fitted_hmodel

        self.coef_ = (
            hierarchical_coef_in_feature_order(
                fitted_hmodel,
                levels,
            )
        )

    def predict(self, X):

        return self.fitted_hmodel.predict(X)


partition_model = (
    HierarchicalModelFeatureOrderView(
        hmodel,
        np.asarray(
            featurizer.bit_sizes_
        ),
    )
)


if np.array_equal(
    np.asarray(featurizer.bit_sizes_),
    np.sort(
        np.asarray(
            featurizer.bit_sizes_
        )
    ),
):

    assert np.allclose(
        partition_model.coef_,
        np.asarray(
            hmodel.coef_
        ).reshape(-1),
    )


# =============================================================================
# HIERARCHICAL MODEL PREDICTIONS
# =============================================================================

y_train_pred = hmodel.predict(X_train)
y_test_pred = hmodel.predict(X_test)


# -------------------------------------------------------------------------
# NEW: Save predictions to CSV
# -------------------------------------------------------------------------

prediction_df = pd.concat(
    [
        pd.DataFrame(
            {
                "smiles": train["smiles"].values,
                "split": "train",
                "e_homo": y_train,
                "prediction": y_train_pred,
                "residual": (
                    y_train - y_train_pred
                ),
            }
        ),

        pd.DataFrame(
            {
                "smiles": test["smiles"].values,
                "split": "test",
                "e_homo": y_test,
                "prediction": y_test_pred,
                "residual": (
                    y_test - y_test_pred
                ),
            }
        ),
    ],
    ignore_index=True,
)


prediction_path = (
    csv_dir / "buhito_e_homo_predictions.csv"
)

if prediction_path.exists():
    print("\nExisting prediction CSV found. Loading:")
    print(" ", prediction_path)
    prediction_df = pd.read_csv(prediction_path)
else:
    prediction_df.to_csv(
        prediction_path,
        index=False,
    )
    print("\nSaved predictions:")
    print(" ", prediction_path)


# =============================================================================
# SAVE HIERARCHICAL MODEL COEFFICIENTS
# =============================================================================

coef_df = pd.DataFrame(
    {
        "bit_id": featurizer.bit_ids_,
        "graphlet_level": featurizer.bit_sizes_,
        "coefficient": partition_model.coef_,
    }
)


coef_path = (
    csv_dir / "buhito_e_homo_coefficients.csv"
)

if coef_path.exists():
    print("\nExisting coefficient CSV found. Loading:")
    print(" ", coef_path)
    coef_df = pd.read_csv(coef_path)
else:
    coef_df.to_csv(
        coef_path,
        index=False,
    )
    print("\nSaved graphlet coefficients:")
    print(" ", coef_path)



# -------------------------------------------------------------------------
# Performance metrics
# -------------------------------------------------------------------------

train_r2 = r2_score(
    y_train,
    y_train_pred,
)

test_r2 = r2_score(
    y_test,
    y_test_pred,
)

train_mae = mean_absolute_error(
    y_train,
    y_train_pred,
)

test_mae = mean_absolute_error(
    y_test,
    y_test_pred,
)

train_rmse = np.sqrt(
    mean_squared_error(
        y_train,
        y_train_pred,
    )
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_test_pred,
    )
)


print("=" * 60)
print("BUHITO HIERARCHICAL MODEL PERFORMANCE")
print("=" * 60)

print(
    f"R^2 Score:\n"
    f"  Train: {train_r2:.4f}\n"
    f"  Test:  {test_r2:.4f}"
)

print(
    f"\nMean Absolute Error (MAE):\n"
    f"  Train: {train_mae:.2f} eV\n"
    f"  Test:  {test_mae:.2f} eV"
)

print(
    f"\nRoot Mean Squared Error (RMSE):\n"
    f"  Train: {train_rmse:.2f} eV\n"
    f"  Test:  {test_rmse:.2f} eV"
)


# =============================================================================
# HIERARCHICAL MODEL PARITY PLOT
# =============================================================================

parity_plot_train_test(
    [X_train, X_test],
    [y_train, y_test],
    hmodel,
    figsize=(12, 4),
    xlab="DFT e_HOMO (eV)",
    ylab="Buhito Hierarchical Model e_HOMO (eV)",
    title="Buhito Hierarchical Residual Model Prediction",
)

plt.savefig(
    images_dir / "hmodel_perf_buhito_e_homo.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()


# =============================================================================
# HIERARCHICAL MODEL COEFFICIENT PLOT
# =============================================================================

fig, ax = plt.subplots(
    figsize=(6, 4),
    dpi=150,
)

hmodel.plot_coefs()

plt.ylabel(
    "Log |beta|",
    fontsize=12,
)

plt.xlabel(
    "Graphlet Level",
    fontsize=12,
)

plt.title(
    "Buhito Hierarchical Model Coefficients by Level",
    fontsize=13,
)

plt.tight_layout()

plt.savefig(
    images_dir / "hmodel_coefs_buhito_e_homo.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()


# =============================================================================
# SAVE METADATA
# =============================================================================

# This records the settings used to produce the fitted model.

metadata = {
    "target": "e_homo",
    "target_description": (
        "Highest Occupied Molecular Orbital Energy"
    ),
    "target_units": "eV",

    "buhito_max_len": BUHITO_MAX_LEN,

    "train_size": len(train),
    "test_size": len(test),

    "train_fraction": 0.8,
    "random_state": 42,

    "ridge_alpha": 1e-5,
    "fit_intercept": False,

    "n_features": int(X_train.shape[1]),

    "train_r2": float(train_r2),
    "test_r2": float(test_r2),

    "train_mae": float(train_mae),
    "test_mae": float(test_mae),

    "train_rmse": float(train_rmse),
    "test_rmse": float(test_rmse),
}


metadata_path = (
    models_dir / "buhito_e_homo_metadata.pkl"
)

if metadata_path.exists():
    print("\nExisting model metadata found. Keeping:")
    print(" ", metadata_path)
else:
    joblib.dump(
        metadata,
        metadata_path,
    )
    print("\nSaved model metadata:")
    print(" ", metadata_path)


# =============================================================================
# 5. BOND-LEVEL PROJECTIONS WITH BUHITO GRAPHLETDAG
# =============================================================================

smiles_strings = [
    "CC",
    "C=C",
    "C#C",
]

mols = [
    Chem.AddHs(
        Chem.MolFromSmiles(s)
    )
    for s in smiles_strings
]

projection_graphs = [
    smiles_to_buhito_graph(
        s,
        add_hs=True,
        output_2d_pos=False,
    )
    for s in smiles_strings
]


dags = []

for mol, graph, smiles in zip(
    mols,
    projection_graphs,
    smiles_strings,
):

    dag = make_buhito_dag(
        mol,
        graph=graph,
        coefficients=partition_model.coef_,
    )

    dags.append(dag)

    print(
        f"Created Buhito DAG for {smiles}: "
        f"{mol.GetNumAtoms()} atoms, "
        f"{mol.GetNumBonds()} bonds"
    )


draw_projected_coefs(
    dags,
    level=2,
    figsize=(12, 12),
    ncol=len(dags),
    cbar_label="Bond-level contribution (eV)",
)

plt.tight_layout()

plt.savefig(
    images_dir / "hmodel_interp_buhito_e_homo.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()


# =============================================================================
# VERIFY DAG PROJECTIONS
# =============================================================================

featurizer.n_jobs = 1
featurizer.verbose = False

model_preds = hmodel.predict(
    featurizer.transform(
        projection_graphs
    )
)

dag_preds = [
    dag.project_to_layer(1).sum()
    for dag in dags
]


print(
    "Verification that Buhito DAG projections "
    "match model predictions:"
)

for smiles, model_pred, dag_pred in zip(
    smiles_strings,
    model_preds,
    dag_preds,
):

    print(
        f"  {smiles:6s}: "
        f"Model={model_pred:8.3f}, "
        f"DAG={dag_pred:8.3f}, "
        f"Diff={abs(model_pred - dag_pred):.6f}"
    )


# =============================================================================
# FINAL OUTPUT SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("\nCSV files:")
print(" ", prediction_path)
print(" ", coef_path)
print(" ", split_path)

print("\nModel files:")
print(" ", featurizer_path)
print(" ", ridge_model_path)
print(" ", hmodel_path)
print(" ", metadata_path)

print("\nFeature files:")
print(" ", X_train_path)
print(" ", X_test_path)

print("\nImage files are saved to:")
print(" ", images_dir)

# =============================================================================
# 6. PARTITION-BASED INTERPRETABILITY
# =============================================================================

import importlib
import partition_interpretability as partition_interpretability

importlib.reload(partition_interpretability)

from partition_interpretability import (
    PartitionInterpreter,
    PartitionOptimizer,
    ChemicalPartitioner,
    visualize_partition_comparison,
    partition_connectivity_penalty,
    connectivity_adjusted_score
)
print("=" * 80)
print("PARTITION-BASED INTERPRETABILITY FOR QM9 MOLECULES")
print("Using Buhito Graphlets Directly")
print("=" * 80)
interpreter_full = PartitionInterpreter(
    featurizer,
    partition_model,
    remove_single_atom_contributions=False
)

interpreter_no_atoms = PartitionInterpreter(
    featurizer,
    partition_model,
    remove_single_atom_contributions=True
)

# Main notebook mode:
interpreter = interpreter_full

# Buhito PartitionInterpreter requires each RDKit molecule to be paired with
# its already-generated NetworkX graph before decomposition.
def register_dataframe_graphs(interpreter_obj, frame):
    for mol, graph in zip(frame["mol"], frame["graph"]):
        interpreter_obj.register_graph(mol, graph)

register_dataframe_graphs(interpreter_full, test)
register_dataframe_graphs(interpreter_no_atoms, test)
print("Registered all test-set Buhito graphs with both interpreters")

print("\n✓ PartitionInterpreter initialized")
print("  Backend: Buhito")
print(f"  Graphlet types: {len(featurizer.bit_ids_)}")
print(f"  Model coefficients: {len(hmodel.coef_)}")
print(f"  Max graphlet size: {BUHITO_MAX_LEN}")
atom_indices = sorted(
    interpreter_no_atoms.atom_level_indices
)

print(
    "Single-atom feature count:",
    len(atom_indices)
)

print(
    "No-atom coefficients are zero:",
    np.allclose(
        interpreter_no_atoms.coefficients[
            atom_indices
        ],
        0.0
    )
)
test_reset = test.reset_index(drop=True)

# Select molecules of different sizes
small_idx = test_reset[test_reset['mol'].apply(lambda m: m.GetNumAtoms() < 8)].index[0]
medium_idx = test_reset[test_reset['mol'].apply(lambda m: 8 <= m.GetNumAtoms() < 12)].index[0]
large_idx = test_reset[test_reset['mol'].apply(lambda m: m.GetNumAtoms() >= 12)].index[0]

selected_mols = [
    (test_reset.loc[small_idx, 'mol'], test_reset.loc[small_idx, 'smiles'], 'Small'),
    (test_reset.loc[medium_idx, 'mol'], test_reset.loc[medium_idx, 'smiles'], 'Medium'),
    (test_reset.loc[large_idx, 'mol'], test_reset.loc[large_idx, 'smiles'], 'Large')
]

print("=" * 80)
print("SELECTED MOLECULES FOR DETAILED PARTITION ANALYSIS")
print("=" * 80)

for mol, smiles, size_label in selected_mols:
    print(f"\n[{size_label}] {smiles}")
    print(f"  Atoms: {mol.GetNumAtoms()}, Bonds: {mol.GetNumBonds()}")
    pred = hmodel.predict(transform_rdkit_mols([mol]))[0]
    print(f"  Predicted ΔH_at: {pred:.2f} eV")
    
    actual_idx = test_reset[test_reset['smiles'] == smiles].index
    if len(actual_idx) > 0:
        actual = test_reset.loc[actual_idx[0], 'e_homo']
        print(f"  Actual e_homo: {actual:.2f} eV")
        print(f"  Error: {abs(pred - actual):.2f} eV")
### Compare Partitioning Strategies

# Test different ways to partition molecules:
# 1. **Functional Groups** - Chemistry-aware (e.g., C=O, NH₂)
# 2. **Distance-based** - Clustering by graph distance
# 3. **Random** - Baseline control

# **Goal:** Find which strategy minimizes higher-order interactions (maximizes interpretability score)
from partition_interpretability import (
    PartitionInterpreter,
    PartitionOptimizer,
    ChemicalPartitioner,
    visualize_partition_comparison
)

from typing import Dict
import matplotlib.pyplot as plt


def compare_with_greedy(
    mol,
    interpreter,
    smiles="",
    k_values=(2, 3, 4),
    greedy_iters=100,
    connectivity_weight=0.10,
    disconnected_singleton_weight=2.0,
    verbose=True
):
    """
    Compare baseline and optimized partition methods using both:

    1. Raw graphlet interpretability score
    2. Connectivity-adjusted score

    adjusted_score =
        raw_score
        - connectivity_weight * connectivity_penalty
    """
    
    if verbose and smiles:
        print(f"\nComparing strategies for: {smiles}")
        print(f"  Testing k values: {list(k_values)}")
        print(f"  Greedy iterations: {greedy_iters}")
        print(
            f"  Connectivity weight: "
            f"{connectivity_weight:.3f}"
        )

    results = []

    def store_result(
        strategy,
        method,
        k,
        interpretation
    ):
        """Compute and store all metrics for one partition result."""

        partition = interpretation.partition
        breakdown = interpretation.get_contribution_breakdown()

        connectivity_penalty = partition_connectivity_penalty(
            mol=mol,
            partition=partition,
            disconnected_singleton_weight=(
                disconnected_singleton_weight
            )
        )

        adjusted_score = connectivity_adjusted_score(
            mol=mol,
            interpretation=interpretation,
            connectivity_weight=connectivity_weight,
            disconnected_singleton_weight=(
                disconnected_singleton_weight
            )
        )
        result = {
            "Strategy": strategy,
            "Method": method,
            "k": k,
            "N_partitions": len(
                set(partition.values())
            ),

            # Original graphlet score
            "Score": interpretation.score,

            # Connectivity terms
            "Connectivity Penalty": connectivity_penalty,
            "Adjusted Score": adjusted_score,

            "Within %": 100 * breakdown["within_frac"],
            "Between %": 100 * breakdown["between_frac"],
            "Higher %": 100 * breakdown["higher_frac"],
            "Interpretation": interpretation
        }

        results.append(result)

        if verbose:
            print(
                f"  {strategy}: "
                f"raw={interpretation.score:.4f}, "
                f"penalty={connectivity_penalty:.4f}, "
                f"adjusted={adjusted_score:.4f}"
            )

    
    # =========================================================================
    # 1. DISTANCE BASELINE
    # =========================================================================

    for k in k_values:
        partition = ChemicalPartitioner.distance_partition(
            mol,
            n_clusters=k,
            seed=42
        )

        interpretation = (
            interpreter.compute_partition_contributions(
                mol,
                partition
            )
        )

        store_result(
            strategy=f"Distance (k={k})",
            method="Distance",
            k=k,
            interpretation=interpretation
        )

    # =========================================================================
    # 2. RANDOM BASELINE
    # =========================================================================

    for k in k_values:
        partition = ChemicalPartitioner.random_partition(
            mol,
            n_clusters=k,
            seed=42
        )

        interpretation = (
            interpreter.compute_partition_contributions(
                mol,
                partition
            )
        )

        store_result(
            strategy=f"Random (k={k})",
            method="Random",
            k=k,
            interpretation=interpretation
        )

    # =========================================================================
    # 3. GREEDY-RANDOM
    # =========================================================================

    for k in k_values:
        optimizer_k = PartitionOptimizer(
            interpreter=interpreter,
            min_partitions=k,
            max_partitions=k,
            connectivity_weight=connectivity_weight,
            disconnected_singleton_weight=(
                disconnected_singleton_weight
            )
        )

        best_interpretation = (
            optimizer_k.optimize_partition_random(
                mol=mol,
                n_iterations=greedy_iters,
                seed=42,
                verbose=False
            )
        )

        store_result(
            strategy=f"Greedy-Random (k={k})",
            method="Greedy-Random",
            k=k,
            interpretation=best_interpretation
        )

    # =========================================================================
    # 4. GREEDY-DISTANCE
    # =========================================================================

    for k in k_values:
        optimizer_k = PartitionOptimizer(
            interpreter=interpreter,
            min_partitions=k,
            max_partitions=k,
            connectivity_weight=connectivity_weight,
            disconnected_singleton_weight=(
                disconnected_singleton_weight
            )
        )

        best_interpretation = (
            optimizer_k.optimize_partition_distance(
                mol=mol,
                n_clusters=k,
                n_iterations=greedy_iters,
                seed=42,
                verbose=False
            )
        )

        store_result(
            strategy=f"Greedy-Distance (k={k})",
            method="Greedy-Distance",
            k=k,
            interpretation=best_interpretation
        )

    # =========================================================================
    # SORT BY CONNECTIVITY-ADJUSTED SCORE
    # =========================================================================

    results.sort(
        key=lambda result: result["Adjusted Score"],
        reverse=True
    )

    if verbose:
        print("\n  Top strategies by adjusted score:")

        for rank, result in enumerate(
            results[:5],
            start=1
        ):
            marker = "🏆" if rank == 1 else f"{rank}."

            print(
                f"    {marker} "
                f"{result['Strategy']:25s} "
                f"raw={result['Score']:.4f}, "
                f"penalty="
                f"{result['Connectivity Penalty']:.4f}, "
                f"adjusted="
                f"{result['Adjusted Score']:.4f}"
            )

        greedy_random_results = [
            result
            for result in results
            if result["Method"] == "Greedy-Random"
        ]

        greedy_distance_results = [
            result
            for result in results
            if result["Method"] == "Greedy-Distance"
        ]

        if (
            greedy_random_results
            and greedy_distance_results
        ):
            best_greedy_random = max(
                greedy_random_results,
                key=lambda result:
                result["Adjusted Score"]
            )

            best_greedy_distance = max(
                greedy_distance_results,
                key=lambda result:
                result["Adjusted Score"]
            )

            print("\n  Greedy strategy comparison:")

            print(
                "    Best Greedy-Random:   "
                f"k={best_greedy_random['k']}, "
                f"adjusted="
                f"{best_greedy_random['Adjusted Score']:.4f}"
            )

            print(
                "    Best Greedy-Distance: "
                f"k={best_greedy_distance['k']}, "
                f"adjusted="
                f"{best_greedy_distance['Adjusted Score']:.4f}"
            )

    return results

def optimize_partition_random_wrapper(
    optimizer,
    mol,
    n_clusters: int,
    n_iterations: int = 100,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[int, int]:
    """
    Run random-start greedy optimization with exactly n_clusters.
    """
    temp_optimizer = PartitionOptimizer(
        interpreter=optimizer.interpreter,
        min_partitions=n_clusters,
        max_partitions=n_clusters,
        connectivity_weight=optimizer.connectivity_weight,
        disconnected_singleton_weight=optimizer.disconnected_singleton_weight
    )

    best_interp = temp_optimizer.optimize_partition_random(
        mol=mol,
        n_iterations=n_iterations,
        seed=seed,
        verbose=verbose,
    )

    return best_interp.partition


def optimize_partition_distance_wrapper(
    optimizer,
    mol,
    n_clusters: int,
    n_iterations: int = 100,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[int, int]:
    """
    Run distance-start greedy optimization with exactly n_clusters.
    """
    temp_optimizer = PartitionOptimizer(
        interpreter=optimizer.interpreter,
        min_partitions=n_clusters,
        max_partitions=n_clusters,
        connectivity_weight=optimizer.connectivity_weight,
        disconnected_singleton_weight=optimizer.disconnected_singleton_weight
    )

    best_interp = temp_optimizer.optimize_partition_distance(
        mol=mol,
        n_clusters=n_clusters,
        n_iterations=n_iterations,
        seed=seed,
        verbose=verbose,
    )

    return best_interp.partition

# ═══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════

all_results = {}

# Define k values to test 
k_values = [2]  

print(f"\n{'═'*80}")
print(f"PARTITION COMPARISON: BASELINES vs GREEDY VARIANTS")
print(f"{'═'*80}")
print(f"\nStrategies tested:")
print(f"  1. Distance baseline (graph distance clustering)")
print(f"  2. Random baseline (random assignment)")
print(f"  3. Greedy-Random (random init + greedy optimization)")
print(f"  4. Greedy-Distance (distance init + greedy optimization)")
print(f"  'greedy algorithms that evolve a random starting partition'")
print(f"  'seeking a partition into regions with minimal interactions'")
print(f"{'═'*80}\n")

for mol, smiles, size_label in selected_mols:
    print(f"\n{'=' * 80}")
    print(f"ANALYZING {size_label} MOLECULE: {smiles}")
    print(f"{'=' * 80}")
    
    # All strategies (Distance, Random, Greedy-Random, Greedy-Distance) use k_values consistently
    results = compare_with_greedy(
        mol=mol,
        interpreter=interpreter,
        smiles=smiles,
        k_values=k_values,
        greedy_iters=100,
        connectivity_weight=0.10,
        disconnected_singleton_weight=2.0,
        verbose=True
    )
    
    all_results[size_label] = results
    
    # Visualize comparison (now includes both greedy strategies)
    fig = visualize_partition_comparison(mol, results, figsize=(20, 12))
    plt.suptitle(f'{size_label} Molecule: {smiles}\n'
                 f'Baselines vs Greedy-Random vs Greedy-Distance', 
                 fontsize=14, fontweight='bold', y=1.00)
    # plt.savefig(images_dir /f'partition_comparison_{size_label.lower()}_both_greedy_e_homo.png', 
    #             dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Partition compariso displayed")


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY ACROSS ALL MOLECULES
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*80}")
print(f"SUMMARY ACROSS ALL MOLECULES")
print(f"{'═'*80}\n")

summary_table = []
for size_label, results in all_results.items():
    best_result = results[0]  # Already sorted by score
    
    # Find best of each type
    best_distance = max([r for r in results if r['Method'] == 'Distance'], 
                        key=lambda x: x['Adjusted Score'])
    best_random = max([r for r in results if r['Method'] == 'Random'], 
                      key=lambda x: x['Adjusted Score'])
    best_greedy_random = max([r for r in results if r['Method'] == 'Greedy-Random'], 
                             key=lambda x: x['Adjusted Score'])
    best_greedy_distance = max([r for r in results if r['Method'] == 'Greedy-Distance'], 
                               key=lambda x: x['Adjusted Score'])
    
    summary_table.append({
        "Molecule": size_label,
        "Overall Winner": best_result["Strategy"],
        "Best Raw Score": best_result["Score"],
        "Best Adjusted Score": best_result["Adjusted Score"],
        "Connectivity Penalty": (
            best_result["Connectivity Penalty"]
        )
    })

import pandas as pd
summary_df = pd.DataFrame(summary_table)
print(summary_df.to_string(index=False))

print(f"\n{'─'*80}")
print(f"KEY FINDINGS:")
print(f"{'─'*80}")

# Count wins by method
winners = [results[0]['Method'] for results in all_results.values()]
from collections import Counter
win_counts = Counter(winners)

print(f"\nWins by Method:")
for method, count in win_counts.most_common():
    print(f"  {method:20s}: {count}/{len(all_results)} molecules")

# Average improvement
print(f"\nAverage Adjusted Scores by Method:")
method_scores = {'Distance': [], 'Random': [], 'Greedy-Random': [], 'Greedy-Distance': []}
for results in all_results.values():
    for result in results:
        method_scores[result['Method']].append(result['Adjusted Score'])

for method in ['Distance', 'Random', 'Greedy-Random', 'Greedy-Distance']:
    scores = method_scores[method]
    if scores:
        mean_score = sum(scores) / len(scores)
        print(f"  {method:20s}: {mean_score:.4f}")

print(f"\n{'═'*80}\n")

from partition_visualization import draw_projected_coefs_with_partitions

# ═══════════════════════════════════════════════════════════════════════════
# GENERAL K-PARTITION VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

K_TO_PLOT = 2

for mol, smiles, size_label in selected_mols:
    print(f"\n{'=' * 80}")
    print(f"{size_label} MOLECULE: {smiles}")
    print(f"{'=' * 80}")

    results = all_results[size_label]

    # Keep only results that actually have K_TO_PLOT partitions
    selected_results = [
        r for r in results
        if r["N_partitions"] == K_TO_PLOT
    ]

    if not selected_results:
        print(f"⚠️ No results found with exactly k={K_TO_PLOT} partitions")
        continue

    selected_results.sort(
        key=lambda x: x["Adjusted Score"],
        reverse=True
    )
    selected_results = selected_results[:6]

    print(f"\nFound {len(selected_results)} methods with exactly k={K_TO_PLOT} partitions")

    dags = []
    partitions = []
    within_contribs = []
    titles = []

    for result in selected_results:
        interp = result["Interpretation"]
        partition = interp.partition
        partition_ids = sorted(set(partition.values()))

        dag = make_buhito_dag(mol, coefficients=interpreter.coefficients)

        dags.append(dag)
        partitions.append(partition)
        within_contribs.append(interp.within_partition)

        within_sum = sum(interp.within_partition.values())
        between_sum = sum(interp.between_partition.values())
        higher_sum = sum(interp.higher_order.values())
        f_G = within_sum + between_sum + higher_sum

        title = f"{result['Strategy']}\n"
        title += f"Actual k={len(partition_ids)} | Score={result['Score']:.4f}\n"
        F0 = interp.within_partition.get(0, 0.0)
        F1 = interp.within_partition.get(1, 0.0)
        F01 = sum(interp.between_partition.values())
        F_higher = sum(interp.higher_order.values())
        f_G = F0 + F1 + F01 + F_higher

        title += f"F₀={F0:+.1f} | F₁={F1:+.1f} | F₀₁={F01:+.1f}\n"
        title += f"f(G)={f_G:+.1f}"

        model_pred = interp.total_prediction

        title += (
            f"\nPred={model_pred:+.1f} | "
            f"Err={abs(f_G - model_pred):.1f}"
        )

        titles.append(title)

    # Atom-level visualization
    fig, axes = draw_projected_coefs_with_partitions(
        graphlet_dags=dags,
        partitions=partitions,
        within_partition_contribs=within_contribs,
        level=1,
        ncol=3,
        titles=titles,
        show_coefficients=True,
        show_bond_labels=False,
        cmap_name="distinct",
        figsize=(18, 6 * ((len(selected_results) + 2) // 3)),
        dpi=150,
        img_size=(600, 600),
        highlight_radius=0.5
    )

    plt.suptitle(
        f"{size_label} Molecule: {smiles}\n"
        f"K={K_TO_PLOT} Partition Visualization — Atom Level",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(images_dir /
        f"k{K_TO_PLOT}_partition_viz_atom_{size_label.lower()}_e_homo.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    # Bond-level visualization
    dags_bond = [
        make_buhito_dag(mol, coefficients=interpreter.coefficients)
        for _ in selected_results
    ]

    fig, axes = draw_projected_coefs_with_partitions(
        graphlet_dags=dags_bond,
        partitions=partitions,
        within_partition_contribs=within_contribs,
        level=2,
        ncol=3,
        titles=titles,
        show_coefficients=True,
        show_bond_labels=True,
        label_precision=2,
        cmap_name="distinct",
        figsize=(18, 6 * ((len(selected_results) + 2) // 3)),
        dpi=150,
        img_size=(700, 700),
        highlight_radius=0.5
    )

    plt.suptitle(
        f"{size_label} Molecule: {smiles}\n"
        f"K={K_TO_PLOT} Partition Visualization — Bond Level",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(images_dir /
        f"k{K_TO_PLOT}_partition_viz_bond_{size_label.lower()}_e_homo.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    print("\nPartition details:")
    for result in selected_results:
        interp = result["Interpretation"]
        partition = interp.partition
        partition_ids = sorted(set(partition.values()))

        print(f"\n{result['Strategy']}")
        print(f"Actual partitions: {partition_ids}")

        for p_id in partition_ids:
            atoms_in_p = [
                idx for idx, part in partition.items()
                if part == p_id
            ]
            symbols = [
                mol.GetAtomWithIdx(idx).GetSymbol()
                for idx in atoms_in_p
            ]

            print(
                f"  P{p_id}: atoms={atoms_in_p}, "
                f"symbols={symbols}, "
                f"F_{p_id}={interp.within_partition.get(p_id, 0.0):+.3f}"
            )

        if interp.between_partition:
            print("  Pair terms:")
            for pair, val in sorted(interp.between_partition.items()):
                print(f"    F_{pair} = {val:+.3f}")

        if interp.higher_order:
            print("  Higher-order terms:")
            for parts, val in sorted(interp.higher_order.items()):
                print(f"    F_{parts} = {val:+.3f}")
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io

PARTITION_COLORS = {
    0: (0.35, 0.55, 1.00),
    1: (1.00, 0.35, 0.35),
    2: (0.35, 0.80, 0.45),
    3: (0.80, 0.55, 1.00),
}

def mol_to_partition_image(mol, partition, img_size=(500, 500)):
    drawer = rdMolDraw2D.MolDraw2DCairo(img_size[0], img_size[1])

    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    opts.fillHighlights = True
    opts.highlightRadius = 0.35
    opts.bondLineWidth = 3

    atoms_to_highlight = []
    atom_colors = {}

    for atom_idx, pid in partition.items():
        atom_idx = int(atom_idx)
        atoms_to_highlight.append(atom_idx)
        atom_colors[atom_idx] = PARTITION_COLORS.get(pid, (0.75, 0.75, 0.75))

    bonds_to_highlight = []
    bond_colors = {}

    for bond in mol.GetBonds():
        bond_idx = bond.GetIdx()
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()

        p1 = partition.get(a1)
        p2 = partition.get(a2)

        bonds_to_highlight.append(bond_idx)

        if p1 == p2:
            bond_colors[bond_idx] = PARTITION_COLORS.get(p1, (0.75, 0.75, 0.75))
        else:
            bond_colors[bond_idx] = (0.60, 0.60, 0.60)

    drawer.DrawMolecule(
        mol,
        highlightAtoms=atoms_to_highlight,
        highlightAtomColors=atom_colors,
        highlightBonds=bonds_to_highlight,
        highlightBondColors=bond_colors
    )

    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def draw_rdkit_partition_grid(
    mol,
    partitions,
    titles,
    mols_per_row=3,
    img_size=(500, 500),
    title_fontsize=10,
    figsize=None
):
    n = len(partitions)
    n_rows = (n + mols_per_row - 1) // mols_per_row

    if figsize is None:
        figsize = (18, 6 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        mols_per_row,
        figsize=figsize
    )

    axes = np.array(axes).reshape(-1)

    for ax, partition, title in zip(axes, partitions, titles):
        img = mol_to_partition_image(
            mol,
            partition,
            img_size=img_size
        )

        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            title,
            fontsize=title_fontsize,
            fontweight="bold",
            pad=10
        )

    for ax in axes[n:]:
        ax.axis("off")

    return fig


K_TO_PLOT = 2

for mol, smiles, size_label in selected_mols:
    print(f"\n{'=' * 80}")
    print(f"{size_label} MOLECULE: {smiles}")
    print(f"{'=' * 80}")

    results = all_results[size_label]

    selected_results = [
        r for r in results
        if r["N_partitions"] == K_TO_PLOT
    ]

    if not selected_results:
        print(f"⚠️ No results found with exactly k={K_TO_PLOT} partitions")
        continue

    selected_results.sort(key=lambda x: x["Score"], reverse=True)
    selected_results = selected_results[:6]

    print(
        f"\nFound {len(selected_results)} methods "
        f"with exactly k={K_TO_PLOT} partitions"
    )

    partitions = []
    titles = []

    for result in selected_results:
        interp = result["Interpretation"]
        partition = interp.partition
        partition_ids = sorted(set(partition.values()))

        partitions.append(partition)

        F0 = interp.within_partition.get(0, 0.0)
        F1 = interp.within_partition.get(1, 0.0)
        F01 = sum(interp.between_partition.values())
        F_higher = sum(interp.higher_order.values())
        f_G = F0 + F1 + F01 + F_higher

        title = f"{result['Strategy']}\n"
        title += f"Actual k={len(partition_ids)} | Score={result['Score']:.4f}\n"
        title += f"F₀={F0:+.1f} | F₁={F1:+.1f} | F₀₁={F01:+.1f}\n"
        title += f"f(G)={f_G:+.1f}"

        model_pred = interp.total_prediction

        title += (
            f"\nPred={model_pred:+.1f} | "
            f"Err={abs(f_G - model_pred):.1f}"
        )

        titles.append(title)

    fig = draw_rdkit_partition_grid(
        mol,
        partitions,
        titles,
        mols_per_row=3,
        img_size=(500, 500),
        title_fontsize=10,
        figsize=(18, 6 * ((len(selected_results) + 2) // 3))
    )

    plt.suptitle(
        f"{size_label} Molecule: {smiles}\n"
        f"K={K_TO_PLOT} Partition Visualization — All Contributions",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    plt.savefig(images_dir /
        f"k{K_TO_PLOT}_partition_viz_rdkit_{size_label.lower()}_e_homo.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("\nPartition details:")

    for result in selected_results:
        interp = result["Interpretation"]
        partition = interp.partition
        partition_ids = sorted(set(partition.values()))

        print(f"\n{result['Strategy']}")
        print(f"Actual partitions: {partition_ids}")

        for p_id in partition_ids:
            atoms_in_p = [
                idx for idx, part in partition.items()
                if part == p_id
            ]

            symbols = [
                mol.GetAtomWithIdx(idx).GetSymbol()
                for idx in atoms_in_p
            ]

            print(
                f"  P{p_id}: atoms={atoms_in_p}, "
                f"symbols={symbols}, "
                f"F_{p_id}={interp.within_partition.get(p_id, 0.0):+.3f}"
            )

        if interp.between_partition:
            print("  Pair terms:")
            for pair, val in sorted(interp.between_partition.items()):
                print(f"    F_{pair} = {val:+.3f}")

        if interp.higher_order:
            print("  Higher-order terms:")
            for parts, val in sorted(interp.higher_order.items()):
                print(f"    F_{parts} = {val:+.3f}")
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io

from partition_interpretability import (
    PartitionOptimizer,
    partition_connectivity_penalty,
    connectivity_adjusted_score
)

CONNECTIVITY_WEIGHT = 0.10

optimizer_k = PartitionOptimizer(
    interpreter=interpreter,
    min_partitions=K_TO_PLOT,
    max_partitions=K_TO_PLOT,
    connectivity_weight=CONNECTIVITY_WEIGHT
)

PARTITION_COLORS = {
    0: (0.35, 0.55, 1.00),
    1: (1.00, 0.35, 0.35),
    2: (0.35, 0.80, 0.45),
    3: (0.80, 0.55, 1.00),
}

def mol_to_partition_image(mol, partition, img_size=(500, 500)):
    drawer = rdMolDraw2D.MolDraw2DCairo(img_size[0], img_size[1])

    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    opts.fillHighlights = True
    opts.highlightRadius = 0.35
    opts.bondLineWidth = 3

    atoms_to_highlight = []
    atom_colors = {}

    for atom_idx, pid in partition.items():
        atom_idx = int(atom_idx)
        atoms_to_highlight.append(atom_idx)
        atom_colors[atom_idx] = PARTITION_COLORS.get(pid, (0.75, 0.75, 0.75))

    bonds_to_highlight = []
    bond_colors = {}

    for bond in mol.GetBonds():
        bond_idx = bond.GetIdx()
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()

        p1 = partition.get(a1)
        p2 = partition.get(a2)

        bonds_to_highlight.append(bond_idx)

        if p1 == p2:
            bond_colors[bond_idx] = PARTITION_COLORS.get(p1, (0.75, 0.75, 0.75))
        else:
            bond_colors[bond_idx] = (0.60, 0.60, 0.60)

    drawer.DrawMolecule(
        mol,
        highlightAtoms=atoms_to_highlight,
        highlightAtomColors=atom_colors,
        highlightBonds=bonds_to_highlight,
        highlightBondColors=bond_colors
    )

    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def draw_rdkit_partition_grid(
    mol,
    partitions,
    titles,
    mols_per_row=3,
    img_size=(500, 500),
    title_fontsize=10,
    figsize=None
):
    n = len(partitions)
    n_rows = (n + mols_per_row - 1) // mols_per_row

    if figsize is None:
        figsize = (18, 6 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        mols_per_row,
        figsize=figsize
    )

    axes = np.array(axes).reshape(-1)

    for ax, partition, title in zip(axes, partitions, titles):
        img = mol_to_partition_image(
            mol,
            partition,
            img_size=img_size
        )

        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            title,
            fontsize=title_fontsize,
            fontweight="bold",
            pad=10
        )

    for ax in axes[n:]:
        ax.axis("off")

    return fig


K_TO_PLOT = 2

CONNECTIVITY_WEIGHT = 0.10

MH_ITERATIONS = 5000
MH_BETA = 20.0
MH_INITIALIZATION = "distance"
MH_SEED = 42

optimizer_k = PartitionOptimizer(
    interpreter=interpreter,
    min_partitions=K_TO_PLOT,
    max_partitions=K_TO_PLOT,
    connectivity_weight=CONNECTIVITY_WEIGHT
)

for mol, smiles, size_label in selected_mols:
    print(f"\n{'=' * 80}")
    print(f"{size_label} MOLECULE: {smiles}")
    print(f"{'=' * 80}")

    # Make a copy so rerunning this cell does not permanently append
    # duplicate Metropolis-Hastings results to all_results.
    results = list(all_results[size_label])

    for result in results:
        interp = result["Interpretation"]

        if "Connectivity Penalty" not in result:
            result["Connectivity Penalty"] = (
                partition_connectivity_penalty(
                    mol,
                    interp.partition
                )
            )

        if "Adjusted Score" not in result:
            result["Adjusted Score"] = (
                connectivity_adjusted_score(
                    mol=mol,
                    interpretation=interp,
                    connectivity_weight=CONNECTIVITY_WEIGHT
                )
            )

    # =====================================================================
    # ADD METROPOLIS-HASTINGS RESULT
    # =====================================================================
    print(
        f"\nRunning Metropolis-Hastings with "
        f"k={K_TO_PLOT}, beta={MH_BETA}, "
        f"iterations={MH_ITERATIONS}"
    )

    mh_interp = (
        optimizer_k.optimize_partition_metropolis_hastings(
            mol=mol,
            n_clusters=K_TO_PLOT,
            n_iterations=MH_ITERATIONS,
            beta=MH_BETA,
            initialization=MH_INITIALIZATION,
            seed=MH_SEED,
            verbose=False
        )
    )

    mh_connectivity_penalty = partition_connectivity_penalty(
        mol,
        mh_interp.partition
    )

    mh_adjusted_score = connectivity_adjusted_score(
        mol=mol,
        interpretation=mh_interp,
        connectivity_weight=CONNECTIVITY_WEIGHT
    )

    mh_result = {
        "Strategy": "Metropolis-Hastings",
        "Method": "Metropolis-Hastings",
        "k": K_TO_PLOT,
        "Score": mh_interp.score,
        "Connectivity Penalty": mh_connectivity_penalty,
        "Adjusted Score": mh_adjusted_score,
        "N_partitions": len(
            set(mh_interp.partition.values())
        ),
        "Interpretation": mh_interp
    }

    results.append(mh_result)

    print(
        f"Metropolis-Hastings result: "
        f"k={mh_result['N_partitions']}, "
        f"raw={mh_result['Score']:.4f}, "
        f"penalty={mh_result['Connectivity Penalty']:.4f}, "
        f"adjusted={mh_result['Adjusted Score']:.4f}"
    )

    # =====================================================================
    # KEEP ONLY RESULTS WITH THE REQUESTED NUMBER OF PARTITIONS
    # =====================================================================
    selected_results = [
        result
        for result in results
        if result["N_partitions"] == K_TO_PLOT
    ]

    if not selected_results:
        print(
            f"⚠️ No results found with exactly "
            f"k={K_TO_PLOT} partitions"
        )
        continue

    # =====================================================================
    # GUARANTEE THAT METROPOLIS-HASTINGS IS SHOWN
    #
    # Keep the five best non-MH methods and always include MH.
    # This produces at most six panels.
    # =====================================================================
    mh_results = [
        result
        for result in selected_results
        if result["Strategy"] == "Metropolis-Hastings"
    ]

    non_mh_results = [
        result
        for result in selected_results
        if result["Strategy"] != "Metropolis-Hastings"
    ]

    non_mh_results.sort(
        key=lambda result: result["Adjusted Score"],
        reverse=True
    )

    selected_results = non_mh_results[:5] + mh_results[:1]

    # Sort the final six methods by score while retaining MH.
    selected_results.sort(
        key=lambda result: result["Adjusted Score"],
        reverse=True
    )

    print(
        f"\nPlotting {len(selected_results)} methods "
        f"with exactly k={K_TO_PLOT} partitions"
    )

    for result in selected_results:
        print(
            f"  {result['Strategy']}: "
            f"raw={result['Score']:.4f}, "
            f"penalty={result['Connectivity Penalty']:.4f}, "
            f"adjusted={result['Adjusted Score']:.4f}"
        )

    partitions = []
    titles = []

    # =====================================================================
    # BUILD PANEL TITLES
    # =====================================================================
    for result in selected_results:
        interp = result["Interpretation"]
        partition = interp.partition
        partition_ids = sorted(
            set(partition.values())
        )

        partitions.append(partition)

        # For k=2, these correspond to the two within-partition terms.
        F0 = interp.within_partition.get(0, 0.0)
        F1 = interp.within_partition.get(1, 0.0)

        # Sum all graphlets spanning two partitions.
        F01 = sum(
            interp.between_partition.values()
        )

        # Sum graphlets spanning three or more partitions.
        F_higher = sum(
            interp.higher_order.values()
        )

        # Reconstructed model prediction.
        f_G = F0 + F1 + F01 + F_higher

        model_pred = interp.total_prediction
        decomposition_error = abs(
            f_G - model_pred
        )

        title = (
            f"{result['Strategy']}\n"
            f"Actual k={len(partition_ids)}\n"
            f"Raw={result['Score']:.4f} | "
            f"Adjusted={result['Adjusted Score']:.4f}\n"
            f"Connectivity penalty="
            f"{result['Connectivity Penalty']:.3f}\n"
            f"F₀={F0:+.1f} | "
            f"F₁={F1:+.1f} | "
            f"F₀₁={F01:+.1f}\n"
            f"f(G)={f_G:+.1f}\n"
            f"Pred={model_pred:+.1f} | "
            f"Err={decomposition_error:.1f}"
        )

        titles.append(title)

    # =====================================================================
    # DRAW PARTITION GRID
    # =====================================================================
    n_rows = (
        len(selected_results) + 2
    ) // 3

    fig = draw_rdkit_partition_grid(
        mol=mol,
        partitions=partitions,
        titles=titles,
        mols_per_row=3,
        img_size=(500, 500),
        title_fontsize=10,
        figsize=(18, 6 * n_rows)
    )

    plt.suptitle(
        f"{size_label} Molecule: {smiles}\n"
        f"K={K_TO_PLOT} Partition Visualization — "
        f"All Contributions",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(
        rect=[0, 0.03, 1, 0.96]
    )

    plt.savefig(images_dir /
        (
            f"k{K_TO_PLOT}_partition_viz_rdkit_"
            f"{size_label.lower()}_mh_e_homo.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # =====================================================================
    # PRINT PARTITION DETAILS
    # =====================================================================
    print("\nPartition details:")

    for result in selected_results:
        interp = result["Interpretation"]
        partition = interp.partition
        partition_ids = sorted(
            set(partition.values())
        )

        print(
            f"\n{result['Strategy']}"
        )
        print(
            f"Score: {result['Score']:.6f}"
        )
        print(
            f"Actual partitions: {partition_ids}"
        )

        for partition_id in partition_ids:
            atoms_in_partition = [
                atom_idx
                for atom_idx, assigned_partition
                in partition.items()
                if assigned_partition == partition_id
            ]

            symbols = [
                mol.GetAtomWithIdx(atom_idx).GetSymbol()
                for atom_idx in atoms_in_partition
            ]

            within_contribution = (
                interp.within_partition.get(
                    partition_id,
                    0.0
                )
            )

            print(
                f"  P{partition_id}: "
                f"atoms={atoms_in_partition}, "
                f"symbols={symbols}, "
                f"F_{partition_id}="
                f"{within_contribution:+.3f}"
            )

        if interp.between_partition:
            print("  Pair terms:")

            for pair, value in sorted(
                interp.between_partition.items()
            ):
                print(
                    f"    F_{pair} = {value:+.3f}"
                )

        if interp.higher_order:
            print("  Higher-order terms:")

            for parts, value in sorted(
                interp.higher_order.items()
            ):
                print(
                    f"    F_{parts} = {value:+.3f}"
                )
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io
from matplotlib.patches import Patch