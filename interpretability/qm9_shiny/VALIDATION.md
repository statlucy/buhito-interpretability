# Validation completed

- Python compilation passed for all supplied and new modules.
- `python check_scoring.py` passed: raw-score extremes, absolute aggregation of partition terms, disconnected singleton weighting, connected singleton exemption, adjusted score, and negative-score ordering without clipping. These checks execute the actual definitions extracted from the supplied module, with a lightweight molecule adjacency fixture.
- Score-panel figure rendered and was visually inspected with synthetic data and a placeholder molecule image. Charts, formulas, labels, and legend fit within the figure. This did not exercise RDKit molecule drawing.
- Hierarchical coefficient alignment, model filenames, explicit hydrogen preprocessing, and MH argument names were checked against the supplied notebook and module.

# Checks requiring your environment

Shiny, RDKit, Buhito, and trained model pickle files were unavailable in the creation environment. Installing Shiny/RDKit from the available package index failed. Thus actual server startup, browser interaction, molecule drawing, model inference, and optimizer integration have not been executed here.

In your working notebook environment, install dependencies, supply the saved e_LUMO artifacts, and launch `shiny run --reload app.py`. Verify:

1. `COC12CC1OC2CO` produces k=2,3,4,5 tabs and a PNG/CSV download.
2. For identical settings, app prediction, raw score, penalty, adjusted score, and returned partition agree with the notebook. Both use the same supplied MH implementation.
3. All partition terms reconstruct the full model prediction within tolerance.
4. Invalid SMILES, disconnected fragments, missing models, and k larger than the explicit atom count produce readable errors.
5. A second analysis clears old results and updates plots/downloads to the submitted molecule and settings.
6. Two browser sessions queue computations without sharing mutable interpreter caches.

## HOMO/LUMO selector update

`python check_properties.py` passed using synthetic loader fixtures: correct model filenames, HOMO → LUMO → HOMO switching, repeated-property cache reuse, no fallback to LUMO when HOMO is missing, optional per-property feature directories, and invalid-property rejection. No trained artifacts were used. Compilation and the existing scoring checks also passed. Full prediction/browser checks still require your environment and both saved model artifacts.

## Caching update

Compilation, scoring checks, and property-routing checks passed. Property-routing fixtures verify one shared featurizer load and reuse of both models across HOMO → LUMO → HOMO. `check_caches.py` executes the analysis cache logic with controlled molecule/model/optimizer fixtures: equivalent canonical-input keys and reordered k values hit; cache hits skip graph construction, transformation, and search; every search setting and target is isolated; artifact metadata changes invalidate results; returned mutation does not corrupt cached results; bounded eviction and clearing work; unseeded requests bypass caching. Canonicalization equivalence is supplied by a fixture because RDKit is unavailable here. Full scientific inference and browser verification still require your local environment and saved artifacts.

Unclipped-score update: removed raw and adjusted clamps; verified negative adjusted scores remain negative and increasing penalties preserve their ordering. Compilation, property routing, and result caching tests passed. Updated plot formulas and negative-score axis limits. Full model/browser execution still requires the local environment.

## Fast analysis update

Compilation and existing scoring, artifact-routing, and cache tests passed. `check_time_budget.py` executes the actual MH method with fixtures: zero budget returns the initial best partition with zero completed proposals; unlimited execution completes all requested proposals. Analysis fixtures verify timeout before search and reporting of partially completed searches without caching them. No 30-second performance claim has been benchmarked with the actual model. Browser startup, full inference, and real molecule drawing still require the user's environment.

## Preloading / preprocessing update

Compilation, scoring, property-routing, completed-result cache, and time-budget tests passed. check_preprocessing.py passed with controlled fixtures: worker preload attempts both properties and shares one featurizer load; canonical molecule caching avoids repeated graph preparation; repeated contexts avoid enumeration and lookup rebuilding; property switching shares only structural feature data while preserving distinct coefficient-weighted instances and predictions; the actual prime_features method copies mutable bit-info; cache sizes remain bounded; artifact replacement invalidates dependent preprocessing; missing one model is reported while retaining the ready property. Full Windows/Shiny browser behavior and actual model inference/performance remain untested here because those dependencies and trained artifacts are not available.
