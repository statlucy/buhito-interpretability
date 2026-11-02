"""Test cooperative MH stopping without the trained model or RDKit."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional
from time import perf_counter
import numpy as np

module=ast.parse(Path(__file__).with_name('partition_interpretability.py').read_text())
cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='PartitionOptimizer')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='optimize_partition_metropolis_hastings')
class Mol:
    def GetNumAtoms(self):return 4
    def GetNumBonds(self):return 1
    def GetBonds(self):return [SimpleNamespace(GetBeginAtomIdx=lambda:0,GetEndAtomIdx=lambda:1)]
ns=dict(np=np,perf_counter=perf_counter,Optional=Optional,Dict=Dict,
    Chem=SimpleNamespace(Mol=Mol),PartitionInterpretation=object,
    partition_connectivity_penalty=lambda **kwargs:.25)
exec(compile(ast.Module(body=[method],type_ignores=[]),'mh','exec'),ns)
interp=SimpleNamespace(partition={0:0,1:0,2:1,3:1},score=.8)
optimizer=SimpleNamespace(interpreter=SimpleNamespace(compute_partition_contributions=lambda *a:interp),
    max_partitions=2,connectivity_weight=.1,disconnected_singleton_weight=2.)
kwargs=dict(mol=Mol(),n_clusters=2,n_iterations=20,seed=42,
            initial_partition=interp.partition,verbose=False)
r=ns[method.name](optimizer,**kwargs,time_budget_seconds=0)
assert r is interp and r.mh_stopped_early and r.mh_iterations_completed==0
r=ns[method.name](optimizer,**kwargs)
assert not r.mh_stopped_early and r.mh_iterations_completed==20
print('Passed: zero-budget best-initial-state return and full fixed-iteration run without deadline.')

# Test app-level analysis budget behavior using the controlled fixtures from check_caches.
from check_caches import ns as analysis_ns
from tempfile import TemporaryDirectory
with TemporaryDirectory() as tmp:
    for name in ['buhito_featurizer.pkl','buhito_hierarchical_model_e_lumo.pkl']:
        (Path(tmp)/name).write_bytes(b'fixture')
    args=['CCO',[2],1000,20.,'distance',42,.1,2.,tmp,'e_lumo']
    try:analysis_ns['analyze'](*args,time_budget_seconds=0)
    except TimeoutError:pass
    else:raise AssertionError('Budget exhausted before search must report timeout.')
    class LimitedOptimizer:
        def __init__(self,**kwargs):pass
        def optimize_partition_metropolis_hastings(self,**kwargs):
            assert kwargs['time_budget_seconds']>0
            return SimpleNamespace(partition={0:0,1:0,2:1,3:1},within_partition={0:20.},
                between_partition={},higher_order={},score=1.,mh_iterations_completed=5,
                mh_stopped_early=True,get_contribution_breakdown=lambda:{'within':20.,'between':0.,'higher':0.})
    analysis_ns['PartitionOptimizer']=LimitedOptimizer
    analysis_ns['clear_caches']()
    p=analysis_ns['analyze'](*args,time_budget_seconds=25)
    assert not p['complete'] and not p['cache_hit']
    assert p['results'][0]['iterations_completed']==5
    assert not analysis_ns['_RESULTS'], 'Partial time-limited results must not be cached.'
print('Passed: setup timeout, partial-result reporting, and exclusion of partial results from cache.')
