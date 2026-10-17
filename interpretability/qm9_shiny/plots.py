"""Molecule drawing and score panels use the notebook's exact definitions."""
import io
import numpy as np
from matplotlib.figure import Figure
from PIL import Image
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
    fig = Figure(figsize=(12,7.6), layout='constrained', facecolor='white')
    grid = fig.add_gridspec(3,2, width_ratios=[1.15,1])
    ax = fig.add_subplot(grid[:,0])
    ax.imshow(mol_to_partition_image(Chem.Mol(payload['mol']),r['partition']))
    ax.axis('off')
    ax.set_title(f"e_LUMO | Metropolis–Hastings | k={r['k']}",fontweight='bold',fontsize=14)
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
    for j,v in enumerate(fractions): ax1.text(v+.02,j,f'{v:.4f}',va='center')
    ax2 = fig.add_subplot(grid[1,1])
    deduction = payload['settings']['weight']*r['penalty']
    vals = [r['raw'],deduction,r['adjusted']]
    ax2.barh(['Raw score','λ × penalty','Adjusted score'],vals,color=['#4f87d6','#d86060','#459c75'])
    ax2.set_xlim(0,1.19)
    ax2.set_title('Raw score → connectivity adjustment',fontweight='bold')
    for j,v in enumerate(vals): ax2.text(v+.02,j,f'{v:.6f}',va='center')
    ax3 = fig.add_subplot(grid[2,1]); ax3.axis('off')
    w,bf,h = fractions
    text = (f'W = {b["within"]:.6g}    B = {b["between"]:.6g}    H = {b["higher"]:.6g}\n'
        'W = Σ|Fₚ|; B = Σ|Fₚq|; H = Σ|Fₛ| for |S| ≥ 3\n'
        'D = W + B + H + 10⁻¹⁰; (w, b, h) = (W, B, H) / D\n'
        f'Raw = clip((w − 0.5b − 2h + 2) / 3, 0, 1) = {r["raw"]:.6f}\n'
        f'Adjusted = clip(Raw − λP, 0, 1) = {r["adjusted"]:.6f}\n'
        f'P = {r["penalty"]:.6f}; λ = {payload["settings"]["weight"]:.3g}\n'
        f'Signed totals: {r["signed"][0]:+.6g}, {r["signed"][1]:+.6g}, {r["signed"][2]:+.6g}\n'
        f'Prediction = {r["prediction"]:.6g}; reconstruction Δ = {r["error"]:.2e}')
    ax3.text(0,1,text,va='top',fontsize=9.5,linespacing=1.5)
    return fig
