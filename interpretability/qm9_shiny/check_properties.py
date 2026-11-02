"""Check property routing and cache isolation without the trained model files."""
import ast
from collections import OrderedDict
from copy import deepcopy
from time import perf_counter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
from properties import property_config, artifact_paths

source = ast.parse(Path(__file__).with_name('analysis.py').read_text())
nodes = [n for n in source.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))
         ]
class Loader:
    def __init__(self):self.calls=[]
    def load(self,path):
        self.calls.append(path.name)
        if path.name=='buhito_featurizer.pkl':
            return SimpleNamespace(bit_sizes_=np.array([1,2]),bit_ids_=[1,2])
        value=10 if path.name=='buhito_hierarchical_model.pkl' else 20
        return SimpleNamespace(unique_levels=[1,2],
            regressors={1:SimpleNamespace(coef_=[value]),2:SimpleNamespace(coef_=[2])},
            predict=lambda X:np.array([value]))
loader=Loader()
ns=dict(Path=Path,np=np,joblib=loader,property_config=property_config,
        artifact_paths=artifact_paths,deepcopy=deepcopy,perf_counter=perf_counter,
        _FEATURIZERS=OrderedDict(),_MODELS=OrderedDict(),_VIEWS=OrderedDict(),
        _RESULTS=OrderedDict(),RESULT_CACHE_SIZE=16,
        _MOLECULES=OrderedDict(),_FEATURES=OrderedDict(),_CONTEXTS=OrderedDict(),
        MOLECULE_CACHE_SIZE=8,FEATURE_CACHE_SIZE=8,CONTEXT_CACHE_SIZE=4)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'loader','exec'),ns)
with TemporaryDirectory() as tmp:
    folder=Path(tmp)
    for name in ['buhito_featurizer.pkl','buhito_hierarchical_model.pkl','buhito_hierarchical_model_e_lumo.pkl']:
        (folder/name).write_bytes(b'fixture')
    load=ns['load_bundle']
    assert load(tmp,'e_homo')[1].predict(None)[0]==10
    assert load(tmp,'e_lumo')[1].predict(None)[0]==20
    count=len(loader.calls)
    load(tmp,'e_lumo');assert len(loader.calls)==count
    assert load(tmp,'e_homo')[1].predict(None)[0]==10
    assert len(loader.calls)==count, 'Property switching must reuse both models and shared featurizer'
    assert loader.calls.count('buhito_featurizer.pkl')==1
    (folder/'buhito_hierarchical_model.pkl').unlink()
    try:load(tmp,'e_homo')
    except FileNotFoundError as exc:assert 'e_HOMO' in str(exc)
    else:raise AssertionError('Missing HOMO must not use LUMO.')
    separate=folder/'e_homo';separate.mkdir()
    assert artifact_paths(folder,'e_homo')[0].parent==separate
    assert artifact_paths(folder,'e_lumo')[0].parent==folder
try:property_config('other')
except ValueError:pass
else:raise AssertionError('Unknown properties must be rejected.')
print('Passed: HOMO/LUMO model routing, switching, cache reuse, missing model, separate feature folders and invalid property.')
