"""Property names and artifact paths, matched to the supplied QM9 notebooks."""
from pathlib import Path

PROPERTIES = {
    'e_lumo': {'label': 'e_LUMO', 'model_file': 'buhito_hierarchical_model_e_lumo.pkl'},
    'e_homo': {'label': 'e_HOMO', 'model_file': 'buhito_hierarchical_model.pkl'},
    "e_at": {
    "label": "E_at (atomization energy, kcal/mol)",
    "model_file": "buhito_hierarchical_model_e_at.pkl",
},
}


def property_config(target):
    if target not in PROPERTIES:
        raise ValueError('Choose e_homo or e_lumo.')
    return PROPERTIES[target]

def artifact_paths(models_dir, target):
    config = property_config(target)
    root = Path(models_dir).resolve()
    # Optional property folders support independently trained feature vocabularies.
    folder = root / target if (root / target).is_dir() else root
    return folder / 'buhito_featurizer.pkl', folder / config['model_file']
