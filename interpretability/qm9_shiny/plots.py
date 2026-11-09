"""Molecule drawing and score panels use the notebook's exact definitions."""
import io
import numpy as np
from matplotlib.figure import Figure
from PIL import Image
from properties import property_config
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

PARTITION_COLORS = {0:(.35,.55,1.), 1:(1.,.35,.35), 2:(.35,.80,.45),
                    3:(.80,.55,1.), 4:(1.,.70,.25)}
CATEGORY_COLORS = ['#4f87d6', '#ef9250', '#a078ca']

def mol_to_partition_image(mol, partition, img_size=(650,650)):
    mol = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(*img_size)
    options = drawer.drawOptions()
    options.useBWAtomPalette()
    options.fillHighlights = True
    options.highlightRadius = .35
    options.bondLineWidth = 3
    atoms = {a:PARTITION_COLORS.get(p,(.75,.75,.75)) for a,p in partition.items()}
    bonds = {}
    for bond in mol.GetBonds():
        p,q = partition[bond.GetBeginAtomIdx()],partition[bond.GetEndAtomIdx()]
        bonds[bond.GetIdx()] = PARTITION_COLORS.get(p,(.75,.75,.75)) if p==q else (.60,.60,.60)
    drawer.DrawMolecule(mol, highlightAtoms=list(atoms), highlightAtomColors=atoms,
                        highlightBonds=list(bonds), highlightBondColors=bonds)
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).copy()

def result_figure(payload, index):
    r = payload['results'][index]
    fig = Figure(figsize=(12,6.2), layout='constrained', facecolor='white')
    grid = fig.add_gridspec(2,2, width_ratios=[1.15,1])
    ax = fig.add_subplot(grid[:,0])
    ax.imshow(mol_to_partition_image(Chem.Mol(payload['mol']),r['partition']))
    ax.axis('off')
    ax.set_title(f"{property_config(payload.get('target', 'e_lumo'))['label']} | Metropolis–Hastings | k={r['k']}" + (' (time limited)' if r.get('stopped_early', False) else ''),fontweight='bold',fontsize=14)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=PARTITION_COLORS[p],label=f'P{p}')
                       for p in sorted(set(r['partition'].values()))],loc='lower center',
              ncol=r['k'],frameon=False)
    categories = ['Within','Between','Higher order']
    b = r['breakdown']
    fractions = [b['within_frac'],b['between_frac'],b['higher_frac']]
    ax1 = fig.add_subplot(grid[0,1])
    ax1.barh(categories,fractions,color=CATEGORY_COLORS)
    ax1.set_xlim(0,1.19)
    ax1.set_xlabel('Fraction of aggregated absolute contributions')
    ax1.set_title('Score inputs: w, b, h',fontweight='bold')
    for j,v in enumerate(fractions): ax1.text(v+.02,j,f'{v:.3g}',va='center')
    ax2 = fig.add_subplot(grid[1,1])
    deduction = payload['settings']['weight']*r['penalty']
    vals = [r['raw'],deduction,r['adjusted']]
    ax2.barh(['Raw score','λ × penalty','Adjusted score'],vals,color=['#4f87d6','#d86060','#459c75'])
    lo, hi = min(0., *vals), max(1., *vals)
    span = hi - lo
    ax2.set_xlim(lo - .24 * span if lo < 0 else 0, hi + .24 * span)
    ax2.axvline(0, color='#777777', linewidth=.7)
    ax2.set_title('Raw score → connectivity adjustment',fontweight='bold')
    for j,v in enumerate(vals):
        ax2.text(v + (.02 * span if v >= 0 else -.02 * span), j, f'{v:.3g}',
                 va='center', ha='left' if v >= 0 else 'right')
    return fig


def mh_diagnostics_figure(payload, index):
    """Histogram of the states occupied by the MH chain for one k value."""
    r = payload['results'][index]
    diagnostics = r.get('mh_diagnostics') or {}
    energy_trace = diagnostics.get('energy_trace', [])
    fig = Figure(figsize=(9, 3.2), layout='constrained', facecolor='white')
    ax = fig.add_subplot(1, 1, 1)
    if energy_trace:
        bins = min(30, max(5, int(np.sqrt(len(energy_trace)))))
        ax.hist(energy_trace, bins=bins, color='#5b8fbc', edgecolor='white')
        ax.axvline(energy_trace[-1], color='#d86060', linewidth=1.5,
                   label='Final state')
        ax.axvline(-diagnostics.get('best_score', r['adjusted']), color='#459c75',
                   linewidth=1.5, linestyle='--', label='Best state')
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(.5, .5, 'No MH trace available', ha='center', va='center',
                transform=ax.transAxes)
    ax.set_title(f"Histogram of MH states visited (k = {r['k']})", fontweight='bold')
    ax.set_xlabel('Energy: E = − adjusted score (dimensionless)')
    ax.set_ylabel('Frequency (iterations)')
    return fig
