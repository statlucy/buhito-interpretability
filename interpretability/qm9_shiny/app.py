"""Run with: shiny run --reload app.py"""
import os
import io
import csv
import asyncio
import multiprocessing
import math
from numbers import Real
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from shiny import App, reactive, render, ui, req
from plots import result_figure, mh_diagnostics_figure
from properties import PROPERTIES, property_config
from worker_tasks import preload, run_analysis

ROOT = Path(__file__).resolve().parent
MODEL_DIR = os.environ.get('QM9_MODELS_DIR', str(ROOT/'models'))
# Spawn avoids inheriting a running event loop. One worker isolates mutable bit-info.
POOL = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'))

_PRELOAD_FUTURE = None

def ensure_preload_future(retry=False):
    """All browser sessions await one shared preload in the same analysis worker."""
    global _PRELOAD_FUTURE
    if _PRELOAD_FUTURE is None or (retry and _PRELOAD_FUTURE.done()):
        _PRELOAD_FUTURE = POOL.submit(preload, MODEL_DIR)
    return _PRELOAD_FUTURE

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h3('Molecule & search'),
        ui.input_select('target', 'Property to predict',
            {key: value['label'] for key, value in PROPERTIES.items()}, selected='e_lumo'),
        ui.input_text('smiles','SMILES','COC12CC1OC2CO'),
        ui.input_text('ks', 'Partition numbers (comma-separated)', '2, 3',
            placeholder='For example: 2, 4, 6'),
        ui.p('Enter one or more integer values. Each must be at least 2 and no greater than the molecule’s atom count, including explicit hydrogens.',
             class_='text-muted small'),
        ui.input_numeric('iterations','Maximum MH iterations per k',1000,min=1,max=100000),
        ui.input_numeric('beta','MH beta',20,min=0,max=1000),
        ui.input_select('initialization','Initialization',{'distance':'Distance','random':'Random'}),
        ui.input_numeric('seed','Random seed',42,min=0,max=4294967295),
        ui.input_numeric('weight','Connectivity weight (λ)',.10,min=0,max=10,step=.01),
        ui.input_numeric('singleton','Disconnected singleton weight',2,min=1,max=100,step=.5),
        ui.input_action_button('analyze','Preparing models…', disabled=True),
        ui.input_action_button('retry_preload', 'Retry model preparation'),
        ui.p('Explicit hydrogens are included, matching model training.'),width=330),
    ui.h2('QM9 orbital energy partition explorer'),
    ui.p('Metropolis–Hastings partitions with separate score and contribution panels.'),
    ui.p('Scores describe the decomposition, not prediction accuracy.'),
    ui.output_ui('preparation_status'),
    ui.output_ui('status'),
    ui.output_ui('results'),
    ui.tags.details(ui.tags.summary('How the connectivity penalty works'),
        ui.p('Within each partition, keep one largest connected component. For every other component, '
             'add its atom count, or the disconnected singleton weight if it contains one atom. '
             'Divide by the molecule’s total atom count, including explicit hydrogens. '
             'A connected one-atom partition has no penalty.')),
    title='QM9 orbital energy partition explorer', fillable=False,
)

def server(input, output, session):
    completed = reactive.value(None)
    message = reactive.value('Preparing models in the analysis worker…')
    model_report = reactive.value(None)
    preparing = reactive.value(True)
    busy = reactive.value(False)

    @reactive.extended_task
    async def prepare_models(retry):
        try:
            future = ensure_preload_future(retry)
            # A session closing must not cancel preparation shared by other sessions.
            return await asyncio.shield(asyncio.wrap_future(future))
        except Exception as exc:
            return {'ready': [], 'errors': {'startup': f'{type(exc).__name__}: {exc}'}}

    @reactive.effect
    @reactive.event(input.retry_preload, ignore_none=False)
    def start_preparation():
        if busy():
            return
        preparing.set(True)
        prepare_models(input.retry_preload() > 0)

    @reactive.effect
    def preparation_finished():
        model_report.set(prepare_models.result())
        preparing.set(False)
        message.set('Model preparation finished. Choose a property and analyze a molecule.')

    @reactive.effect
    def update_analyze_button():
        report = model_report()
        ready = not preparing() and report is not None and input.target() in report['ready']
        ui.update_action_button('analyze', disabled=not ready or busy(),
            label='Analyzing…' if busy() else 'Analyze molecule' if ready else 'Preparing / model unavailable')
        ui.update_action_button('retry_preload', disabled=busy() or preparing())

    @render.ui
    def preparation_status():
        if preparing():
            return ui.p('Preparing models… This startup step is outside the molecule-analysis timer.')
        report = model_report()
        if report is None:
            return None
        ready = ', '.join(property_config(key)['label'] for key in report['ready']) or 'none'
        lines = [ui.p(f"Models ready: {ready}. Preparation: {report.get('seconds', 0):.1f}s.")]
        lines += [ui.p(f'{key}: {error}', class_='text-danger') for key,error in report['errors'].items()]
        return ui.div(*lines)

    @reactive.extended_task
    async def calculate(args):
        try:
            value = await asyncio.get_running_loop().run_in_executor(POOL, run_analysis, *args)
            return {'value':value}
        except Exception as exc:
            return {'error':f'{type(exc).__name__}: {exc}'}

    @reactive.effect
    @reactive.event(input.analyze)
    def start():
        if busy() or preparing():
            return
        report = model_report()
        if report is None or input.target() not in report['ready']:
            message.set('The selected model is not ready. Check preparation errors and retry.')
            return
        try:
            raw_ks = input.ks().strip()
            try:
                parts = [part.strip() for part in raw_ks.split(',')]
                if not parts or any(not part for part in parts):
                    raise ValueError
                ks = tuple(int(part) for part in parts)
            except ValueError:
                raise ValueError(
                    'Enter one or more comma-separated whole numbers, for example: 2, 4, 6.'
                )
            iterations, seed = input.iterations(),input.seed()
            beta, weight, singleton = input.beta(),input.weight(),input.singleton()
            if not input.smiles().strip() or not ks:
                raise ValueError('Enter a SMILES string and at least one partition number.')
            if any(k < 2 for k in ks):
                raise ValueError('Each partition number must be at least 2.')
            if len(set(ks)) != len(ks):
                raise ValueError('Enter each partition number only once.')
            import math
            nums = [iterations, seed, beta, weight, singleton]
            if any(v is None or not math.isfinite(v) for v in nums):
                raise ValueError('All settings must be finite numbers.')
            if iterations != int(iterations) or not 1 <= iterations <= 100000:
                raise ValueError('Iterations must be an integer from 1 to 100000.')
            if seed != int(seed) or not 0 <= seed <= 4294967295:
                raise ValueError('Seed must be an integer from 0 to 4294967295.')
            if not 0 <= beta <= 1000 or not 0 <= weight <= 10 or not 1 <= singleton <= 100:
                raise ValueError('Search settings are outside their allowed ranges.')
            args = (input.smiles(),ks,int(iterations),float(beta),input.initialization(),
                    int(seed),float(weight),float(singleton),MODEL_DIR,input.target(),None)
            completed.set(None)
            message.set(f'Running {property_config(input.target())["label"]} Metropolis–Hastings… Results will appear when all selected k values finish.')
            busy.set(True)
            calculate(args)
        except Exception as exc:
            completed.set(None)
            message.set(str(exc))

    @reactive.effect
    def finish():
        response = calculate.result()
        busy.set(False)
        if 'error' in response:
            message.set(response['error'])
        else:
            completed.set(response['value'])
            if response['value'].get('cache_hit', False):
                message.set('Loaded matching completed analysis from cache. Results correspond to the submitted settings.')
            else:
                value = response['value']
                if value.get('complete', True):
                    message.set(f"Analysis complete in {value['elapsed_seconds']:.1f}s of worker computation. Results cached.")
                else:
                    message.set(f"Time budget reached after {value['elapsed_seconds']:.1f}s of worker computation. "
                        f"Showing best partitions found; skipped k values: {value.get('skipped_ks', [])}. "
                        "These partial results are not cached.")

    @render.ui
    def status():
        return ui.div(message(),class_='alert alert-info',role='status')

    @render.ui
    def results():
        payload = completed()
        if payload is None:
            return ui.p('Required for this property: buhito_featurizer.pkl and ' +
                        property_config(input.target())['model_file'] + '. Select settings and click Analyze molecule.')
        panels = [ui.nav_panel(f"k = {r['k']}",
                              ui.output_plot(f'plot_{r["k"]}', height='760px'),
                              ui.h4('Metropolis–Hastings diagnostics'),
                              ui.output_ui(f'diagnostics_{r["k"]}'),
                              ui.output_plot(f'mh_plot_{r["k"]}', height='340px'),
                              ui.download_button(f'png_{r["k"]}','Download PNG'))
                  for r in payload['results']]
        return ui.div(
            ui.h4(f"Predicted {payload['property_label']}: {payload['prediction']:.6g} (dataset units)"),
            ui.p('SMILES: '+payload['smiles']),
            ui.p('Worker stage times (seconds; original run for cached results): ' +
                 ', '.join(f'{name}: {value:.2f}' for name,value in payload.get('timings', {}).items())),
            ui.p('Reused preprocessing: ' + (', '.join(
                name for name,hit in payload.get('preprocessing_cache', {}).items() if hit) or 'none')),
            ui.navset_tab(*panels),
            ui.h4('Result summary'),ui.output_data_frame('summary'),
            ui.download_button('csv_download','Download summary CSV'),
            ui.download_button('atom_download','Download atom partitions CSV'))

    # Register fixed output IDs once; only selected tabs are displayed.
    def register_k(k):
        @render.ui
        def diagnostics():
            payload = completed()
            req(payload is not None)
            result = next((r for r in payload['results'] if r['k'] == k), None)
            req(result is not None)
            d = result.get('mh_diagnostics') or {}
            if not d:
                return ui.p('MH diagnostics are unavailable for this result.', class_='text-muted')
            invalid = d['same_partition_proposals'] + d['singleton_source_proposals']
            return ui.div(
                ui.h5('MH diagnostics'),
                ui.p(
                    f"β = {d['beta']:.3g} inverse adjusted-score units "
                    "(the adjusted score and energy are dimensionless)."
                ),
                ui.tags.ul(
                    ui.tags.li(
                        f"Accepted moves: {d['accepted_moves']} "
                        f"({d['acceptance_rate']:.1%} of {d['valid_proposals']} valid proposals)"
                    ),
                    ui.tags.li(f"Rejected valid moves: {d['rejected_valid_moves']}"),
                    ui.tags.li(
                        f"Invalid proposals: {invalid} ({d['same_partition_proposals']} same-partition; "
                        f"{d['singleton_source_proposals']} would empty a partition)"
                    ),
                    ui.tags.li(
                        f"Accepted improvements: {d['accepted_improving_moves']}; "
                        f"accepted worse moves: {d['accepted_worse_moves']}"
                    )
                ),
                class_='mb-2'
            )
        output(id=f'diagnostics_{k}')(diagnostics)

        @render.plot
        def mh_plot():
            payload = completed()
            req(payload is not None)
            indexes = [i for i, r in enumerate(payload['results']) if r['k'] == k]
            req(bool(indexes))
            return mh_diagnostics_figure(payload, indexes[0])
        output(id=f'mh_plot_{k}')(mh_plot)

        @render.plot
        def plot():
            payload = completed()
            req(payload is not None)
            indexes = [i for i,r in enumerate(payload['results']) if r['k']==k]
            req(bool(indexes))
            return result_figure(payload,indexes[0])
        output(id=f'plot_{k}')(plot)

        @render.download(filename=lambda:f'{completed()["target"]}_mh_k{k}.png')
        def png():
            payload = completed()
            req(payload is not None)
            index = next(i for i,r in enumerate(payload['results']) if r['k']==k)
            buf = io.BytesIO()
            result_figure(payload,index).savefig(buf,format='png',dpi=180)
            yield buf.getvalue()
        output(id=f'png_{k}')(png)

    # A molecule can contain at most 100 explicit atoms in this app, so
    # register every allowable output ID once. Only requested k values are
    # inserted into the UI and evaluated.
    for k in range(2,101): register_k(k)

    def rows(payload):
        return [dict(Iterations_completed=r.get('iterations_completed'),
            Search_seconds=r.get('search_seconds'),
            Property=payload['target'],SMILES=payload['smiles'],k=r['k'],Prediction=r['prediction'],
            Raw=r['raw'],Penalty=r['penalty'],Adjusted=r['adjusted'],
            Within=r['signed'][0],Between=r['signed'][1],Higher=r['signed'][2],
            Reconstruction_error=r['error'],
            Accepted_moves=(r.get('mh_diagnostics') or {}).get('accepted_moves'),
            Rejected_valid_moves=(r.get('mh_diagnostics') or {}).get('rejected_valid_moves'),
            Invalid_same_partition=(r.get('mh_diagnostics') or {}).get('same_partition_proposals'),
            Invalid_singleton_source=(r.get('mh_diagnostics') or {}).get('singleton_source_proposals'),
            Acceptance_rate=(r.get('mh_diagnostics') or {}).get('acceptance_rate'),
            **payload['settings']) for r in payload['results']]

    def format_summary_value(value):
        """Format real-valued table cells to three significant figures only for display."""
        if isinstance(value, Real) and not isinstance(value, (bool, int)) and math.isfinite(value):
            return f'{value:.3g}'
        return value

    def summary_rows(payload):
        return [
            {column: format_summary_value(value) for column, value in row.items()}
            for row in rows(payload)
        ]

    @render.data_frame
    def summary():
        import pandas as pd
        req(completed() is not None)
        return render.DataGrid(pd.DataFrame(summary_rows(completed())),height='250px')

    @render.download(filename=lambda:f'{completed()["target"]}_mh_summary.csv')
    def csv_download():
        req(completed() is not None)
        data=rows(completed()); buf=io.StringIO()
        writer=csv.DictWriter(buf,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
        yield buf.getvalue().encode()

    @render.download(filename=lambda:f'{completed()["target"]}_mh_atom_partitions.csv')
    def atom_download():
        from rdkit import Chem
        payload=completed();req(payload is not None)
        mol=Chem.Mol(payload['mol']);buf=io.StringIO();writer=csv.writer(buf)
        writer.writerow(['property','SMILES','k','atom_index','element','partition'])
        for r in payload['results']:
            for a,p in sorted(r['partition'].items()):
                writer.writerow([payload['target'],payload['smiles'],r['k'],a,mol.GetAtomWithIdx(a).GetSymbol(),p])
        yield buf.getvalue().encode()

app = App(app_ui, server)
app.on_shutdown(lambda: POOL.shutdown(wait=False,cancel_futures=True))
