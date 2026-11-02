"""Analysis"""
from pathlib import Path
from collections import OrderedDict
from copy import deepcopy
from time import perf_counter
import numpy as np
import joblib
from rdkit import Chem
from buhito.converters import smiles_to_nx
from properties import PROPERTIES, property_config, artifact_paths
from partition_interpretability import (PartitionInterpreter, PartitionOptimizer,
    partition_connectivity_penalty, connectivity_adjusted_score)

# Worker-local caches. The single analysis worker serializes access to mutable bit-info.
_FEATURIZERS = OrderedDict()
_MODELS = OrderedDict()
_VIEWS = OrderedDict()
_RESULTS = OrderedDict()
_MOLECULES = OrderedDict()
_FEATURES = OrderedDict()
_CONTEXTS = OrderedDict()
MOLECULE_CACHE_SIZE = 8
FEATURE_CACHE_SIZE = 8
CONTEXT_CACHE_SIZE = 4
RESULT_CACHE_SIZE = 16


def artifact_versions(models_dir, target):
    config = property_config(target)
    paths = artifact_paths(models_dir, target)
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing model files: ' + ', '.join(missing) +
            f'. Place the matching {config["label"]} artifacts in {paths[0].parent}')
    versions = []
    for path in paths:
        stat = path.stat()
        versions.append((str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size))
    return tuple(versions)


def _retain(cache, key, value, limit):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)
    return value


def _load_artifact(cache, version, limit):
    path = version[0]
    entry = cache.get(path)
    if entry is not None and entry[0] == version:
        cache.move_to_end(path)
        return entry[1]
    # Replace stale entries for this path rather than accumulating versions.
    if entry is not None and entry[0] != version:
        # Dispose of stale interpreters and their coefficient copies on artifact replacement.
        _CONTEXTS.clear()
        if cache is _FEATURIZERS:
            _FEATURES.clear()
    value = joblib.load(Path(path))
    _retain(cache, path, (version, value), limit)
    return value


def clear_caches():
    """Release in-memory results and artifacts; restart also clears all caches."""
    for cache in (_RESULTS, _CONTEXTS, _FEATURES, _MOLECULES, _VIEWS, _MODELS, _FEATURIZERS):
        cache.clear()


def canonicalize_smiles(smiles):
    base = Chem.MolFromSmiles(smiles.strip())
    if base is None or base.GetNumAtoms() == 0:
        raise ValueError('Enter a valid, nonempty SMILES string.')
    if len(Chem.GetMolFrags(base)) != 1:
        raise ValueError('Enter one connected molecule; dot-separated fragments are unsupported.')
    return Chem.MolToSmiles(base, canonical=True, isomericSmiles=True)


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

def load_bundle(models_dir, target="e_lumo"):
    versions = artifact_versions(models_dir, target)
    featurizer = _load_artifact(_FEATURIZERS, versions[0], 2)
    featurizer.n_jobs = 1
    featurizer.verbose = False
    model = _load_artifact(_MODELS, versions[1], 4)
    if versions in _VIEWS:
        _VIEWS.move_to_end(versions)
        view = _VIEWS[versions]
    else:
        view = HierarchicalModelFeatureOrderView(model, featurizer.bit_sizes_)
        if len(view.coef_) != len(featurizer.bit_ids_):
            raise ValueError('Model and featurizer feature counts differ.')
        # Drop coefficient views for replaced files, including their old model references.
        paths = {version[0] for version in versions}
        for old in list(_VIEWS):
            if old != versions and any(v[0] in paths and v not in versions for v in old):
                del _VIEWS[old]
        _retain(_VIEWS, versions, view, 4)
    return featurizer, view

def prepare_molecule(smiles):
    canonical = canonicalize_smiles(smiles)
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

def preload_models(models_dir):
    """Runs inside the long-lived worker, before analysis is enabled in the browser."""
    started = perf_counter()
    report = {'ready': [], 'errors': {}}
    for target in PROPERTIES:
        try:
            load_bundle(models_dir, target)
            report['ready'].append(target)
        except Exception as exc:
            report['errors'][target] = f'{type(exc).__name__}: {exc}'
    report['seconds'] = perf_counter() - started
    return report


def load_molecule(canonical):
    if canonical in _MOLECULES:
        _MOLECULES.move_to_end(canonical)
        return _MOLECULES[canonical], True
    value = prepare_molecule(canonical)
    _retain(_MOLECULES, canonical, value, MOLECULE_CACHE_SIZE)
    return value, False


def load_context(canonical, target, versions, mol, graph, featurizer, model):
    """Reuse molecule features across properties; coefficient-weighted instances stay isolated."""
    key = (canonical, target, versions)
    if key in _CONTEXTS:
        _CONTEXTS.move_to_end(key)
        interpreter, prediction = _CONTEXTS[key]
        return interpreter, prediction, True, True
    interpreter = PartitionInterpreter(featurizer, model, remove_single_atom_contributions=False)
    interpreter.register_graph(mol, graph)
    feature_key = (canonical, versions[0])
    feature_hit = feature_key in _FEATURES
    if feature_hit:
        _FEATURES.move_to_end(feature_key)
        fingerprint, bitinfo = _FEATURES[feature_key]
        interpreter.prime_features(mol, fingerprint, bitinfo)
    else:
        fingerprint = interpreter.get_fingerprint(mol)
        bitinfo = deepcopy(interpreter.get_bitinfo(mol))
        _retain(_FEATURES, feature_key, (fingerprint, bitinfo), FEATURE_CACHE_SIZE)
    prediction = float(model.predict(interpreter.get_fingerprint(mol))[0])
    # Populate instance enumeration and atom-to-instance lookup once for all k and MH settings.
    interpreter.enumerate_graphlet_instances(mol, include_zero_coefficients=True)
    interpreter.get_atom_to_instance_indices(mol)
    _retain(_CONTEXTS, key, (interpreter, prediction), CONTEXT_CACHE_SIZE)
    return interpreter, prediction, feature_hit, False


def analyze(smiles, ks, iterations, beta, initialization, seed, weight, singleton_weight, models_dir, target="e_lumo", time_budget_seconds=None):
    started = perf_counter()
    deadline = None if time_budget_seconds is None else started + float(time_budget_seconds)
    timings = {}
    config = property_config(target)
    if not ks:
        raise ValueError('Select at least one k value.')
    canonical = canonicalize_smiles(smiles)
    ks = sorted(set(map(int, ks)))
    versions = artifact_versions(models_dir, target)
    cache_key = (canonical, target, tuple(ks), iterations, beta, initialization,
                 seed, weight, singleton_weight, versions, time_budget_seconds)
    if seed is not None and cache_key in _RESULTS:
        _RESULTS.move_to_end(cache_key)
        payload = deepcopy(_RESULTS[cache_key])
        payload['cache_hit'] = True
        payload['elapsed_seconds'] = perf_counter() - started
        return payload
    (_, mol, graph), molecule_hit = load_molecule(canonical)
    timings["molecule"] = perf_counter() - started
    if any(k < 2 or k > mol.GetNumAtoms() for k in ks):
        raise ValueError(f'Each k must be between 2 and {mol.GetNumAtoms()} (including hydrogens).')
    stage_started = perf_counter()
    featurizer, model = load_bundle(models_dir, target)
    timings["model_loading"] = perf_counter() - stage_started
    stage_started = perf_counter()
    interpreter, prediction, feature_hit, context_hit = load_context(
        canonical, target, versions, mol, graph, featurizer, model)
    timings['features_and_prediction'] = perf_counter() - stage_started
    results = []
    for index, k in enumerate(ks):
        remaining = None if deadline is None else deadline - perf_counter()
        if remaining is not None and remaining <= 0:
            if results:
                break
            raise TimeoutError('The time budget was used by model loading or featurization before search. '
                f'Stage times (seconds): {timings}. Models are now cached; try again with one k.')
        stage_started = perf_counter()
        search_kwargs = {} if remaining is None else {'time_budget_seconds': remaining / (len(ks)-index)}
        optimizer = PartitionOptimizer(interpreter=interpreter, min_partitions=k, max_partitions=k,
            connectivity_weight=weight, disconnected_singleton_weight=singleton_weight)
        interp = optimizer.optimize_partition_metropolis_hastings(mol=mol, n_clusters=k,
            n_iterations=iterations, beta=beta, initialization=initialization,
            seed=seed, verbose=False, **search_kwargs)
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
        results.append(dict(k=k,
            iterations_completed=getattr(interp, 'mh_iterations_completed', iterations),
            stopped_early=getattr(interp, 'mh_stopped_early', False),
            search_seconds=perf_counter()-stage_started, partition=partition, breakdown=breakdown, signed=signed,
            within=dict(interp.within_partition), between=dict(interp.between_partition),
            higher=dict(interp.higher_order), raw=float(interp.score),
            penalty=float(partition_connectivity_penalty(mol, partition, singleton_weight)),
            adjusted=float(connectivity_adjusted_score(mol, interp, weight, singleton_weight)),
            prediction=prediction, reconstructed=reconstructed,
            error=abs(reconstructed-prediction)))
    timings['partition_search'] = sum(r['search_seconds'] for r in results)
    complete = len(results) == len(ks) and all(not r['stopped_early'] for r in results)
    payload = dict(elapsed_seconds=perf_counter()-started, timings=timings,
        complete=complete, preprocessing_cache=dict(molecule=molecule_hit, features=feature_hit, interpreter=context_hit), skipped_ks=[k for k in ks if k not in [r['k'] for r in results]], target=target, property_label=config["label"], cache_hit=False,
        smiles=canonical, mol=mol.ToBinary(), prediction=prediction, results=results,
        settings=dict(iterations=iterations, beta=beta, initialization=initialization, seed=seed,
                      weight=weight, singleton_weight=singleton_weight, time_budget_seconds=time_budget_seconds))
    # Do not cache if files changed during a computation. A new request must reload them.
    if complete and seed is not None and artifact_versions(models_dir, target) == versions:
        _retain(_RESULTS, cache_key, deepcopy(payload), RESULT_CACHE_SIZE)
    return payload
