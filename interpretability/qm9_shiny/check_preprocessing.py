"""Exercise preload and structural/property cache reuse with controlled fixtures."""
from check_properties import ns, loader, TemporaryDirectory, Path, SimpleNamespace
from collections import Counter
from copy import deepcopy
import numpy as np
import ast

calls=Counter()
module=ast.parse(Path(__file__).with_name('partition_interpretability.py').read_text())
cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='PartitionInterpreter')
prime=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='prime_features')
method_ns=dict(deepcopy=deepcopy)
exec(compile(ast.Module(body=[prime],type_ignores=[]),'prime_features','exec'),method_ns)
class Molecule:
    def ToBinary(self):return b'molecule'
class Interpreter:
    prime_features=method_ns['prime_features']
    def __init__(self,feat,model,**kwargs):
        calls['interpreter']+=1
        self.model=model;self.bit_ids=feat.bit_ids_
        self._fingerprint_cache={};self._bitinfo_cache={}
    def _get_molecule_cache_key(self,mol):return mol.ToBinary()
    def register_graph(self,*args):pass
    def get_fingerprint(self,mol):
        key=mol.ToBinary()
        if key not in self._fingerprint_cache:
            calls['transform']+=1
            self._fingerprint_cache[key]=np.array([[1.,0.]])
            self._bitinfo_cache[key]={'locations':[0,1]}
        return self._fingerprint_cache[key]
    def get_bitinfo(self,mol):return self._bitinfo_cache[mol.ToBinary()]
    def enumerate_graphlet_instances(self,*args,**kwargs):
        calls['enumeration']+=1
        self.coefficient=float(self.model.coef_[0])
    def get_atom_to_instance_indices(self,*args):calls['lookup']+=1

def prepare(canonical):
    calls['prepare']+=1
    return canonical,Molecule(),object()
ns.update(PartitionInterpreter=Interpreter,prepare_molecule=prepare)
ns['clear_caches']()
with TemporaryDirectory() as tmp:
    folder=Path(tmp)
    for name in ['buhito_featurizer.pkl','buhito_hierarchical_model.pkl','buhito_hierarchical_model_e_lumo.pkl']:
        (folder/name).write_bytes(b'fixture')
    loader.calls.clear()
    report=ns['preload_models'](tmp)
    assert set(report['ready'])=={'e_homo','e_lumo'} and not report['errors']
    assert loader.calls.count('buhito_featurizer.pkl')==1
    assert len(loader.calls)==3
    (canonical,mol,graph),hit=ns['load_molecule']('CCO');assert not hit
    _,hit=ns['load_molecule']('CCO');assert hit and calls['prepare']==1
    feat,homo=ns['load_bundle'](tmp,'e_homo')
    _,lumo=ns['load_bundle'](tmp,'e_lumo');assert len(loader.calls)==3
    hv=ns['artifact_versions'](tmp,'e_homo');lv=ns['artifact_versions'](tmp,'e_lumo')
    h,pred,fh,ch=ns['load_context'](canonical,'e_homo',hv,mol,graph,feat,homo)
    assert pred==10 and not fh and not ch
    again,_,fh,ch=ns['load_context'](canonical,'e_homo',hv,mol,graph,feat,homo)
    assert again is h and fh and ch and calls['enumeration']==1
    l,pred,fh,ch=ns['load_context'](canonical,'e_lumo',lv,mol,graph,feat,lumo)
    assert pred==20 and fh and not ch
    assert calls['transform']==1 and calls['enumeration']==2 and calls['lookup']==2
    assert h.coefficient==10 and l.coefficient==20
    # Shared location input is copied by the real prime_features method.
    l.get_bitinfo(mol)['locations'].append(99)
    assert 99 not in h.get_bitinfo(mol)['locations']
    assert 99 not in ns['_FEATURES'][(canonical,hv[0])][1]['locations']
    for i in range(10):
        (name,mol,graph),_=ns['load_molecule'](f'mol{i}')
        ns['load_context'](name,'e_homo',hv,mol,graph,feat,homo)
    assert len(ns['_MOLECULES'])<=8 and len(ns['_FEATURES'])<=8 and len(ns['_CONTEXTS'])<=4
    (folder/'buhito_featurizer.pkl').write_bytes(b'changed featurizer')
    ns['load_bundle'](tmp,'e_homo')
    assert not ns['_FEATURES'] and not ns['_CONTEXTS']
    (folder/'buhito_hierarchical_model.pkl').unlink()
    report=ns['preload_models'](tmp)
    assert report['ready']==['e_lumo'] and 'e_homo' in report['errors']
    ns['clear_caches']()
    assert not any(ns[key] for key in ('_MOLECULES','_FEATURES','_CONTEXTS'))
print('Passed: worker preload, graph/feature/context reuse, model isolation, bit-info copy isolation, bounded caches and artifact invalidation.')
