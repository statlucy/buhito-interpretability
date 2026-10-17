"""Inference-only e_LUMO analysis. Called serially by one isolated worker."""
from pathlib import Path
import io
import numpy as np
import joblib
from rdkit import Chem
from buhito.converters import smiles_to_nx
from partition_interpretability import (PartitionInterpreter, PartitionOptimizer,
    partition_connectivity_penalty, connectivity_adjusted_score)

_BUNDLE = None

class HierarchicalModelFeatureOrderView:
    def __init__(self, model, levels):
        self.fitted_hmodel = model
        levels = np.asarray(levels)
        self.coef_ = np.zeros(len(levels), dtype=float)
        for level in model.unique_levels:
            mask = levels == level
            values = np.asarray(model.regressors[level].coef_).reshape(-1)
            if mask.sum() != len(values):
                raise ValueError(f'Coefficient count mismatch at level {level}.')
            self.coef_[mask] = values
    def predict(self, X):
        return self.fitted_hmodel.predict(X)

def load_bundle(models_dir):
    global _BUNDLE
    folder = Path(models_dir).resolve()
    paths = [folder / 'buhito_featurizer.pkl',
             folder / 'buhito_hierarchical_model_e_lumo.pkl']
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing model files: ' + ', '.join(missing) +
                                '. Place the saved e_LUMO artifacts in ' + str(folder))
    key = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
    if _BUNDLE is None or _BUNDLE[0] != key:
        featurizer, model = (joblib.load(p) for p in paths)
        view = HierarchicalModelFeatureOrderView(model, featurizer.bit_sizes_)
        if len(view.coef_) != len(featurizer.bit_ids_):
            raise ValueError('Model and featurizer feature counts differ.')
        _BUNDLE = (key, featurizer, view)
    return _BUNDLE[1:]

def prepare_molecule(smiles):
    smiles = smiles.strip()
    base = Chem.MolFromSmiles(smiles)
    if base is None or base.GetNumAtoms() == 0:
        raise ValueError('Enter a valid, nonempty SMILES string.')
    if len(Chem.GetMolFrags(base)) != 1:
        raise ValueError('Enter one connected molecule; dot-separated fragments are unsupported.')
    # Build BOTH representations from the same canonical SMILES, preserving atom order.
    canonical = Chem.MolToSmiles(base, canonical=True, isomericSmiles=True)
    mol = Chem.AddHs(Chem.MolFromSmiles(canonical))
    if mol.GetNumAtoms() > 100:
        raise ValueError('This app supports at most 100 atoms including explicit hydrogens.')
    converted = smiles_to_nx(canonical, add_hs=True, output_2d_pos=False)
    graph = converted[0] if isinstance(converted, (tuple, list)) else converted
    if set(map(int, graph.nodes)) != set(range(mol.GetNumAtoms())):
        raise ValueError('Buhito graph nodes do not match the RDKit atom indices.')
    if {tuple(sorted((int(a), int(b)))) for a,b in graph.edges} != {
            tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()}:
        raise ValueError('Buhito and RDKit bond indices do not align.')
    return canonical, mol, graph

def analyze(smiles, ks, iterations, beta, initialization, seed, weight, singleton_weight, models_dir):
    if not ks:
        raise ValueError('Select at least one k value.')
    canonical, mol, graph = prepare_molecule(smiles)
    ks = sorted(set(map(int, ks)))
    if any(k < 2 or k > mol.GetNumAtoms() for k in ks):
        raise ValueError(f'Each k must be between 2 and {mol.GetNumAtoms()} (including hydrogens).')
    featurizer, model = load_bundle(models_dir)
    interpreter = PartitionInterpreter(featurizer, model, remove_single_atom_contributions=False)
    interpreter.register_graph(mol, graph)
    prediction = float(model.predict(featurizer.transform([graph]))[0])
    results = []
    for k in ks:
        optimizer = PartitionOptimizer(interpreter=interpreter, min_partitions=k, max_partitions=k,
            connectivity_weight=weight, disconnected_singleton_weight=singleton_weight)
        interp = optimizer.optimize_partition_metropolis_hastings(mol=mol, n_clusters=k,
            n_iterations=iterations, beta=beta, initialization=initialization,
            seed=seed, verbose=False)
        partition = {int(a): int(p) for a,p in interp.partition.items()}
        if set(partition) != set(range(mol.GetNumAtoms())) or len(set(partition.values())) != k:
            raise RuntimeError(f'Optimizer returned an invalid partition for k={k}.')
        breakdown = interp.get_contribution_breakdown()
        signed = [sum(interp.within_partition.values()), sum(interp.between_partition.values()),
                  sum(interp.higher_order.values())]
        reconstructed = float(sum(signed))
        if not np.isclose(reconstructed, prediction, atol=1e-7, rtol=1e-6):
            raise RuntimeError('Partition contributions do not reconstruct the model prediction. '
                               'Check model/featurizer compatibility and preprocessing.')
        results.append(dict(k=k, partition=partition, breakdown=breakdown, signed=signed,
            within=dict(interp.within_partition), between=dict(interp.between_partition),
            higher=dict(interp.higher_order), raw=float(interp.score),
            penalty=float(partition_connectivity_penalty(mol, partition, singleton_weight)),
            adjusted=float(connectivity_adjusted_score(mol, interp, weight, singleton_weight)),
            prediction=prediction, reconstructed=reconstructed,
            error=abs(reconstructed-prediction)))
    return dict(smiles=canonical, mol=mol.ToBinary(), prediction=prediction, results=results,
        settings=dict(iterations=iterations, beta=beta, initialization=initialization, seed=seed,
                      weight=weight, singleton_weight=singleton_weight))
