"""Exercise actual analysis caching with deterministic molecular/model fixtures."""
from check_properties import ns, loader, TemporaryDirectory, Path, SimpleNamespace
from collections import Counter

calls=Counter()
class Molecule:
    def GetNumAtoms(self):return 4
    def ToBinary(self):return b'molecule fixture'
class Interpreter:
    def __init__(self,*args,**kwargs):pass
    def register_graph(self,*args):pass
    def get_fingerprint(self,*args):
        calls['transform']+=1
        return SimpleNamespace()
    def get_bitinfo(self,*args):return {}
    def prime_features(self,*args):pass
    def enumerate_graphlet_instances(self,*args,**kwargs):pass
    def get_atom_to_instance_indices(self,*args):pass
class Optimizer:
    def __init__(self,**kwargs):self.k=kwargs['min_partitions']
    def optimize_partition_metropolis_hastings(self,**kwargs):
        calls['mh']+=1
        return SimpleNamespace(partition={a:a%self.k for a in range(4)},
            within_partition={0:20.},between_partition={},higher_order={},score=1.,
            get_contribution_breakdown=lambda:{'within':20.,'between':0.,'higher':0.})
def prepare(smiles):
    calls['prepare']+=1
    return smiles,Molecule(),object()
def bundle(*args):
    return SimpleNamespace(transform=lambda x:calls.update(['transform']) or x),SimpleNamespace(predict=lambda x:[20.])
ns.update(PartitionInterpreter=Interpreter,PartitionOptimizer=Optimizer,
          prepare_molecule=prepare,canonicalize_smiles=lambda s:'CCO' if s.strip() in ('CCO','OCC') else s.strip(),
          load_bundle=bundle,partition_connectivity_penalty=lambda *a:0.,
          connectivity_adjusted_score=lambda *a:1.)
ns['clear_caches']()
with TemporaryDirectory() as tmp:
    for name in ['buhito_featurizer.pkl','buhito_hierarchical_model.pkl','buhito_hierarchical_model_e_lumo.pkl']:
        (Path(tmp)/name).write_bytes(b'fixture')
    args=['CCO',[2,3],1000,20.,'distance',42,.1,2.,tmp,'e_lumo']
    first=ns['analyze'](*args);assert not first['cache_hit']
    before=calls.copy()
    equivalent=args.copy();equivalent[0]=' OCC ';equivalent[1]=[3,2,2]
    hit=ns['analyze'](*equivalent)
    assert hit['cache_hit'] and calls==before, 'Hit must skip graph, transform and search'
    hit['results'][0]['partition'][0]=99
    assert ns['analyze'](*args)['results'][0]['partition'][0]!=99
    for index,value in [(0,'CCC'),(1,[2]),(2,2000),(3,10.),(4,'random'),(5,43),(6,.2),(7,3.),(9,'e_homo')]:
        changed=args.copy();changed[index]=value
        assert not ns['analyze'](*changed)['cache_hit'], (index,value)
    # The fixture's predictor stays at 20 for both properties; only cache isolation is tested.
    (Path(tmp)/'buhito_hierarchical_model_e_lumo.pkl').write_bytes(b'new model version')
    assert not ns['analyze'](*args)['cache_hit']
    assert ns['analyze'](*args)['cache_hit']
    (Path(tmp)/'buhito_featurizer.pkl').write_bytes(b'new featurizer version')
    assert not ns['analyze'](*args)['cache_hit']
    ns['RESULT_CACHE_SIZE']=2;ns['_RESULTS'].clear()
    for seed in [1,2,3]:
        changed=args.copy();changed[5]=seed;ns['analyze'](*changed)
    assert len(ns['_RESULTS'])==2
    changed=args.copy();changed[5]=1
    assert not ns['analyze'](*changed)['cache_hit'], 'Evicted result must be recomputed'
    changed[5]=None
    assert not ns['analyze'](*changed)['cache_hit']
    assert not ns['analyze'](*changed)['cache_hit'], 'Unseeded requests must remain stochastic'
    ns['clear_caches']();assert not ns['_RESULTS']
print('Passed: equivalent inputs, skipped work, settings/property isolation, artifact invalidation, copy isolation, eviction and cache clearing.')
