# QM9 e_HOMO / e_LUMO Shiny application

Choose e_HOMO or e_LUMO, enter one connected SMILES string and run the supplied Metropolis–Hastings optimizer for selected k values from 2 to 5. Each result tab shows the colored molecule on the left and separate score-input, connectivity-adjustment, and calculation panels on the right. Download individual PNGs, a summary CSV, or atom partition assignments.

## Setup

Use the Python environment in which your e_LUMO notebook and Buhito model already work. Install Shiny and the listed dependencies:

```bash
python -m pip install -r requirements.txt
```

Buhito is a separate prerequisite: install your existing Buhito checkout in that same environment, for example:

```bash
python -m pip install -e /path/to/buhito
```

Copy these **matching saved artifacts from your QM9 notebooks** into this app's `models` directory:

- `buhito_featurizer.pkl`
- `buhito_hierarchical_model_e_lumo.pkl` (LUMO)
- `buhito_hierarchical_model.pkl` (HOMO; this is the exact filename saved by your HOMO notebook)

Only the model for the selected property is required. The trained artifacts are not included in this download. The app does not retrain, and no QM9 CSV or training feature matrices are needed. Use the original training environment's Buhito, scikit-learn, NumPy, and joblib versions to avoid pickle compatibility problems. Only load your own trusted pickle files.

Alternatively set `QM9_MODELS_DIR` to your existing models directory before launching:

```bash
# macOS/Linux
export QM9_MODELS_DIR=/path/to/buhito/models
```

```powershell
# Windows PowerShell
$env:QM9_MODELS_DIR = "C:\path\to\buhito\models"
```

From the extracted application directory, run:

```bash
shiny run --reload app.py
```

Open the local URL printed in the terminal. Stop the server with Ctrl+C.

## Scientific behavior

- Selects the hierarchical HOMO or LUMO model based on the property dropdown; never substitutes the other model if a file is missing.
- Switching property requires clicking Analyze molecule to compute new results. Existing results and download labels retain the property that produced them.
- Both notebook workflows use `buhito_featurizer.pkl`. It must be the fitted featurizer used to train the selected model. If the models were trained with different feature vocabularies, use `models/e_homo/` and `models/e_lumo/`, each holding its own `buhito_featurizer.pkl` and correctly named model. A property subfolder, when present, is used exclusively for that property; missing files there cause an error. Matching feature counts alone cannot prove vocabularies are identical.
- Preserves the notebook's hierarchical coefficient alignment by graphlet level.
- Canonicalizes the input once, then builds the RDKit molecule and Buhito graph from that same string, both with explicit hydrogens. Verifies atom-index sets and bond endpoints agree.
- Every search uses min_partitions=max_partitions=k and checks the returned partition covers every atom with exactly k nonempty groups.
- One seeded run is performed per selected k. The optimizer returns its best encountered partition; global optimality is not guaranteed.
- Uses all graphlet contributions, including single-atom terms.
- W, B, H are sums of absolute values **after aggregation into partition terms**. Signed category totals reconstruct the prediction and are shown separately.
- Raw = (w - 0.5b - 2h + 2)/3, where (w,b,h)=(W,B,H)/(W+B+H+1e-10).
- Connectivity penalty: retain one largest connected component in each partition; penalize other components by their size, or singleton weight for a one-atom component; normalize by all atoms. A connected singleton partition has no penalty.
- Adjusted = Raw - lambda * penalty. This is the optimization objective.
- Displays decomposition discrepancy, not predictive error against experimental data. Units are labeled “dataset units”; confirm the units in your QM9 preprocessing before relabeling.
- Valid SMILES outside QM9 are not a guarantee of reliable predictions. Unknown graphlets and larger molecules can reduce coverage. Inputs must contain one connected component and at most 100 atoms including H; resource limits are configurable in `analysis.py`.

## Files and execution

`app.py` contains the interface. `analysis.py` loads artifacts and performs inference/search. `plots.py` draws the molecule and adjacent score panels. `properties.py` maps each property to its model filename and display label. `partition_interpretability.py` is your supplied implementation with score clipping removed. `partition_visualization.py` is included as a reference; the new renderer lives in `plots.py`.

One spawned worker serializes analyses and retains the model/featurizer once loaded, preventing concurrent mutation of Buhito bit-info. Browser sessions can remain responsive while awaiting computation. Multiple sessions queue behind that worker. Larger models require enough memory for both the app and worker. No models are loaded merely by opening the interface.

## Validation

See `VALIDATION.md` for checks completed in the creation environment and remaining model-dependent checks. Before interpreting results, compare a known notebook molecule under identical k, seed, beta, initialization, and iteration settings with this app.

## Updating an existing installation

Replace **all four files**: `app.py`, `analysis.py`, `plots.py`, and `properties.py`. Keep your existing model files. Stop the current server, then restart with `shiny run --reload app.py`. Updating only app.py is insufficient because the worker and renderer also need the selected property.

## In-memory caching

The single analysis worker now keeps separate caches for fitted featurizers, trained models, and aligned coefficient views. A shared featurizer is loaded once; switching HOMO → LUMO → HOMO retains both models rather than reloading them each time. Separate property featurizers remain isolated by absolute path. Artifact caches are bounded to two featurizers, four models, and four coefficient views.

Completed successful analyses are cached in a least-recently-used cache of 16 results. The key contains canonical isomeric SMILES, property, sorted unique k values, iterations, beta, initialization, seed, connectivity weight, disconnected singleton weight, and the absolute paths/modification time/change time/size of BOTH artifact files. A matching request skips graph conversion, artifact loading, prediction, and MH searches. RDKit still parses and canonicalizes the input, and artifact metadata is checked. Plots are rendered from the saved results when displayed or downloaded.

The status message identifies a cache hit. Result copies prevent one request from modifying cached data. Errors are not cached. Unseeded requests (if called programmatically with seed=None) are not cached. File metadata changes invalidate matching keys automatically; if files are modified while an analysis runs, that result is not cached. These metadata checks are not content hashes: restart the server if artifact bytes were replaced while preserving all file metadata.

Caches live in the worker's memory and are shared across browser sessions served by that worker. They are cleared by restarting the server, automatic reload after code changes, or calling analysis.clear_caches() inside the worker. They do not persist to disk. Retaining both models increases RAM usage, while avoiding repeated disk reads; the existing single worker still serializes analyses. Different molecular atom orderings that canonicalize to the same string share the same cached result.

For this caching update, replace `analysis.py` and `app.py` with the versions in this package. Keep `plots.py` and `properties.py` from the HOMO/LUMO update. No model retraining is needed.

## Unclipped scores

Raw and adjusted scores now return their formulas directly. Raw normalization remains unchanged and naturally gives values approximately between 0 and 1. Adjusted scores may be negative. MH optimization and score reports both use the unclipped adjusted score, so negative candidates remain distinguishable. The score chart supports negative bars. Replace `partition_interpretability.py` and `plots.py` from this package and restart the server to clear old cached results.

## Fast analysis target

The UI defaults to one k (2), 1,000 maximum iterations, and Fast analysis enabled. Fast analysis shares a 25-second worker-computation budget across setup and selected k searches. Each remaining k gets a share of the time left. The MH loop checks a cooperative deadline before each proposal and returns its best encountered state if time expires. Remaining k values may be skipped. Time-limited output is labeled and the table reports actual iterations and search seconds. Stage timings separate molecule preparation, model loading, featurization/prediction, and partition searches.

This is a best-effort target, NOT a guaranteed 30-second completed analysis. A blocking model load, featurization, initial decomposition, single proposal, or rendering operation cannot be interrupted by the cooperative deadline. Worker startup/imports, queue wait, transport, plotting, and browser display are outside the worker-computation timer. Large or cold requests can exceed 30 seconds. If setup consumes the budget, the app reports that instead of claiming a completed analysis; loaded artifacts remain cached for a retry. Only fully completed searches are cached; truncated searches are not reused as completed results.

Time-limited runs may execute different numbers of iterations even with the same seed because machine speed and load vary. For fixed-iteration reproducibility, disable Fast analysis. Existing notebook callers are unchanged because time_budget_seconds defaults to None. The MH loop also avoids recalculating the same connectivity penalty twice per candidate; the unclipped adjusted-score formula is unchanged.

Replace `app.py`, `analysis.py`, `partition_interpretability.py`, and `plots.py` from this package and restart. After testing your usual molecule, read the stage timings to determine whether the 30-second goal needs deeper graphlet optimizations or faster model loading. No hardware runtime benchmark was possible in the creation environment.

## Model preparation and reusable molecule preprocessing

When the first browser session opens, the app submits one shared preparation job to the SAME single worker that executes analyses. It preloads LUMO and HOMO models and aligned coefficients, reusing the shared featurizer. Other sessions await that same future. Analyze remains disabled for a property until preparation reports it ready. If only one model is available, you can select and use that property; the missing model error remains visible. After adding/correcting files, click Retry model preparation. Preparation time is displayed separately from analysis time. No molecule graphlets can be preloaded for an arbitrary unseen molecule before its input is known.

Additional bounded, worker-local LRU caches now retain:

- 8 canonical molecules with explicit-H RDKit molecules and Buhito graphs, independent of search settings or target.
- 8 molecule/featurizer-version feature rows and copied graphlet-location dictionaries. Both properties reuse these only if they use the exact same fitted featurizer artifact.
- 4 molecule/property/artifact-version interpreter contexts, including predictions, coefficient-weighted graphlet instances, and atom-to-instance lookup structures. Changing k, seed, beta, iterations, or connectivity weights reuses this context while running a fresh search. These contexts NEVER share coefficient-weighted instances across properties.

The first context builds features once and reuses that feature row for prediction and decomposition. Property switching can reuse graphlet locations, but computes the correct new property-specific contributions. Bit-info is copied to isolate cached locations from later transformer mutations. Replacing artifact versions invalidates dependent contexts; changing a featurizer also clears feature caches. All caches are cleared by server restart. Retention is bounded but increases RAM usage; lower MOLECULE_CACHE_SIZE, FEATURE_CACHE_SIZE, and CONTEXT_CACHE_SIZE in analysis.py if needed.

For this update replace `app.py`, `analysis.py`, and `partition_interpretability.py`, and add the new `worker_tasks.py` beside them. Keep `plots.py` and `properties.py` from the previous update. Restart with `shiny run --reload app.py`, wait for model preparation, then analyze. Worker entry points live in worker_tasks.py so Windows spawn does not need to import the UI or create another process pool.

Preloading moves loading costs before the Analyze click; it does not eliminate those costs. Unseen-molecule graphlet enumeration and MH search can still exceed 30 seconds. Use the reported stage times to assess actual runtime on your computer. No real-model runtime benchmark was possible in the creation environment.
