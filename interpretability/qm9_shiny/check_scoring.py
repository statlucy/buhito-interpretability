"""Dependency-light checks of the supplied score and connectivity definitions."""
import ast
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, Tuple
import math

source = ast.parse(Path(__file__).with_name('partition_interpretability.py').read_text())
names = {'PartitionInterpretation','partition_connectivity_penalty','connectivity_adjusted_score'}
nodes = [n for n in source.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
class Atom:
    def __init__(self,mol,index):self.mol,self.index=mol,index
    def GetIdx(self):return self.index
    def GetNeighbors(self):return [Atom(self.mol,i) for i in self.mol.neighbors[self.index]]
class Mol:
    def __init__(self,n,edges):
        self.neighbors={i:[] for i in range(n)}
        for a,b in edges:self.neighbors[a].append(b);self.neighbors[b].append(a)
    def GetNumAtoms(self):return len(self.neighbors)
    def GetAtomWithIdx(self,i):return Atom(self,i)
class Chem:pass
Chem.Mol=Mol
ns=dict(dataclass=dataclass,defaultdict=defaultdict,deque=deque,Dict=Dict,Tuple=Tuple,Chem=Chem)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'supplied_scoring','exec'),ns)
Interpretation=ns['PartitionInterpretation'];penalty=ns['partition_connectivity_penalty']
for fields,expected in [(({0:1},{},{}),1),(({}, {(0,1):1},{}),.5),(({}, {},{(0,1,2):1}),0)]:
    r=Interpretation(*fields,total_prediction=1,partition={0:0})
    assert math.isclose(r.score,expected,abs_tol=1e-9)
r=Interpretation({0:2,1:-2},{(0,1):-3},{(0,1,2):1},-2,{0:0})
b=r.get_contribution_breakdown()
assert b['within']==4 and b['between']==3 and b['higher']==1
assert math.isclose(r.score,(.5-.5*.375-2*.125+2)/3,abs_tol=1e-9)
m=Mol(4,[(0,1),(1,2),(2,3)])
assert penalty(m,{0:0,1:0,2:1,3:1})==0
assert penalty(m,{0:0,1:1,2:0,3:1},2)==1
assert penalty(m,{0:0,1:1,2:1,3:1},2)==0
r.partition={0:0,1:1,2:0,3:1}
assert math.isclose(ns['connectivity_adjusted_score'](m,r,.1,2),r.score-.1)
assert math.isclose(ns['connectivity_adjusted_score'](m,r,10,2),r.score-10)
assert ns['connectivity_adjusted_score'](m,r,10,2)<0
assert ns['connectivity_adjusted_score'](m,r,20,2)<ns['connectivity_adjusted_score'](m,r,10,2)
print('Passed: raw score extremes, aggregation, connectivity, singleton, unclipped adjustment and negative-score ordering.')
