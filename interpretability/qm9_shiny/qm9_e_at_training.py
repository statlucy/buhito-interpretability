"""Train QM9 E_at models using the feature cache shared by e_HOMO and e_LUMO.

Run this from the directory that contains ``models``, ``features``, and ``csv``:

    python qm9_e_at_training.py

The script intentionally does not fit a new featurizer.  It requires the
existing ``buhito_featurizer.pkl``, ``X_train_buhito.npz``, and
``X_test_buhito.npz`` so the saved E_at model is compatible with the app's
existing feature order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from buhito.regressors import HierarchicalResidualModel


TARGET_COLUMN = "E_at"
TARGET_KEY = "e_at"
TARGET_LABEL = "E_at (atomization energy, kcal/mol)"


def load_feature_matrix(path: Path, key: str) -> np.ndarray | sparse.spmatrix:
    """Load either the sparse or dense cache format used by the prior scripts."""
    try:
        return sparse.load_npz(path)
    except Exception:
        with np.load(path) as data:
            return data[key]


def first_existing(paths: list[Path], description: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    attempted = "\n  ".join(str(path) for path in paths)
    raise FileNotFoundError(f"Could not find {description}. Checked:\n  {attempted}")


def read_split(data: pd.DataFrame, split_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recreate train/test row order exactly from the cached feature split."""
    split = pd.read_csv(split_path)
    required = {"smiles", "split"}
    if not required.issubset(split.columns):
        raise ValueError(f"{split_path} must contain columns: {sorted(required)}")
    if split["smiles"].duplicated().any() or data["smiles"].duplicated().any():
        raise ValueError("SMILES must be unique to align the saved split with feature rows.")

    indexed = data.set_index("smiles", drop=False)
    train_smiles = split.loc[split["split"] == "train", "smiles"].tolist()
    test_smiles = split.loc[split["split"] == "test", "smiles"].tolist()
    if not train_smiles or not test_smiles:
        raise ValueError(f"{split_path} does not contain both train and test rows.")
    missing = set(train_smiles + test_smiles).difference(indexed.index)
    if missing:
        raise ValueError(f"{split_path} contains {len(missing)} SMILES absent from the QM9 CSV.")
    return indexed.loc[train_smiles].reset_index(drop=True), indexed.loc[test_smiles].reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("../examples/qm9/qm9data/qm9_processed.csv"),
        help="QM9 processed CSV containing the E_at column.",
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--features-dir", type=Path, default=Path("features"))
    parser.add_argument("--csv-dir", type=Path, default=Path("csv"))
    parser.add_argument(
        "--split-path",
        type=Path,
        default=None,
        help="Existing e_LUMO/e_HOMO split that created the cached feature rows.",
    )
    parser.add_argument("--force", action="store_true", help="Retrain and overwrite existing E_at PKLs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    models_dir = args.models_dir.resolve()
    features_dir = args.features_dir.resolve()
    csv_dir = args.csv_dir.resolve()
    data_path = args.data.resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"QM9 data file not found: {data_path}")
    models_dir.mkdir(parents=True, exist_ok=True)

    featurizer_path = models_dir / "buhito_featurizer.pkl"
    if not featurizer_path.exists():
        raise FileNotFoundError(
            f"Shared fitted featurizer not found: {featurizer_path}\n"
            "Create or copy the app's existing buhito_featurizer.pkl before training E_at."
        )

    train_feature_path = first_existing(
        [features_dir / "X_train_buhito.npz", models_dir / "X_train_buhito.npz"],
        "cached X_train_buhito.npz",
    )
    test_feature_path = first_existing(
        [features_dir / "X_test_buhito.npz", models_dir / "X_test_buhito.npz"],
        "cached X_test_buhito.npz",
    )
    split_path = args.split_path.resolve() if args.split_path else first_existing(
        [
            csv_dir / "buhito_e_lumo_train_test_split.csv",
            csv_dir / "buhito_e_homo_train_test_split.csv",
        ],
        "the e_LUMO or e_HOMO train/test split used for the cached feature matrices",
    )

    data = pd.read_csv(data_path)
    if "smiles" not in data.columns or TARGET_COLUMN not in data.columns:
        raise ValueError(f"{data_path} must contain 'smiles' and '{TARGET_COLUMN}' columns.")
    train, test = read_split(data, split_path)

    featurizer = joblib.load(featurizer_path)
    X_train = load_feature_matrix(train_feature_path, "X_train")
    X_test = load_feature_matrix(test_feature_path, "X_test")
    if X_train.shape[0] != len(train) or X_test.shape[0] != len(test):
        raise ValueError(
            "Cached feature rows do not match the selected train/test split. "
            "Pass --split-path for the split that was used to create these matrices."
        )
    if X_train.shape[1] != len(featurizer.bit_ids_) or X_test.shape[1] != len(featurizer.bit_ids_):
        raise ValueError("Feature cache columns do not match buhito_featurizer.pkl.")

    y_train = train[TARGET_COLUMN].to_numpy(dtype=float)
    y_test = test[TARGET_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(y_train).all() or not np.isfinite(y_test).all():
        raise ValueError(f"{TARGET_COLUMN} contains missing or non-finite values.")

    ridge_path = models_dir / "buhito_ridge_model_e_at.pkl"
    hmodel_path = models_dir / "buhito_hierarchical_model_e_at.pkl"
    metadata_path = models_dir / "buhito_e_at_metadata.json"

    if hmodel_path.exists() and not args.force:
        print(f"Existing E_at hierarchical model kept: {hmodel_path}")
        print("Use --force to retrain it.")
        return

    ridge = Ridge(fit_intercept=False, alpha=1e-5, solver="sparse_cg", tol=1e-5)
    ridge.fit(X_train, y_train)
    joblib.dump(ridge, ridge_path)

    hmodel = HierarchicalResidualModel(
        regressor=Ridge(fit_intercept=False, alpha=1e-5, solver="sparse_cg"),
        verbose=1,
    )
    hmodel.fit(X_train, y_train, levels=np.asarray(featurizer.bit_sizes_))
    if len(np.asarray(hmodel.coef_).reshape(-1)) != len(featurizer.bit_ids_):
        raise RuntimeError("Hierarchical coefficient count does not match the app featurizer.")
    joblib.dump(hmodel, hmodel_path)

    prediction = hmodel.predict(X_test)
    metrics = {
        "target_key": TARGET_KEY,
        "target_column": TARGET_COLUMN,
        "label": TARGET_LABEL,
        "units": "kcal/mol",
        "featurizer": featurizer_path.name,
        "train_features": str(train_feature_path),
        "test_features": str(test_feature_path),
        "split": str(split_path),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": int(X_train.shape[1]),
        "test_r2": float(r2_score(y_test, prediction)),
        "test_mae": float(mean_absolute_error(y_test, prediction)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
    }
    metadata_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Saved Ridge model: {ridge_path}")
    print(f"Saved hierarchical model: {hmodel_path}")
    print(f"Saved metadata: {metadata_path}")
    print(
        "Test metrics: "
        f"R²={metrics['test_r2']:.4f}, MAE={metrics['test_mae']:.4f}, "
        f"RMSE={metrics['test_rmse']:.4f} kcal/mol"
    )
    print("Add the e_at entry from properties.py to the app's property configuration.")


if __name__ == "__main__":
    main()
