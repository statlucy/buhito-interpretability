"""Run with: shiny run --reload app.py"""
import os
import io
import csv
import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from shiny import App, reactive, render, ui, req
from plots import result_figure

ROOT = Path(__file__).resolve().parent
MODEL_DIR = os.environ.get('QM9_MODELS_DIR', str(ROOT/'models'))
# Spawn avoids inheriting a running event loop. One worker isolates mutable bit-info.
POOL = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'))

def run_analysis(*args):
    from analysis import analyze
    return analyze(*args)

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h3('Molecule & search'),
        ui.input_text('smiles','SMILES','COC12CC1OC2CO'),
        ui.input_checkbox_group('ks','Number of partitions (k)',
            {str(k):str(k) for k in range(2,6)},selected=['2','3','4','5'],inline=True),
        ui.input_numeric('iterations','MH iterations per k',5000,min=1,max=100000),
        ui.input_numeric('beta','MH beta',20,min=0,max=1000),
        ui.input_select('initialization','Initialization',{'distance':'Distance','random':'Random'}),
        ui.input_numeric('seed','Random seed',42,min=0,max=4294967295),
        ui.input_numeric('weight','Connectivity weight (λ)',.10,min=0,max=10,step=.01),
        ui.input_numeric('singleton','Disconnected singleton weight',2,min=1,max=100,step=.5),
        ui.input_task_button('analyze','Analyze molecule'),
        ui.p('Explicit hydrogens are included, matching model training.'),width=330),
    ui.h2('QM9 e_LUMO partition explorer'),
    ui.p('Metropolis–Hastings partitions with separate score and contribution panels.'),
    ui.p('Scores describe the decomposition, not prediction accuracy.'),
    ui.output_ui('status'),
    ui.output_ui('results'),
    ui.tags.details(ui.tags.summary('How the connectivity penalty works'),
        ui.p('Within each partition, keep one largest connected component. For every other component, '
             'add its atom count, or the disconnected singleton weight if it contains one atom. '
             'Divide by the molecule’s total atom count, including explicit hydrogens. '
             'A connected one-atom partition has no penalty.')),
    title='QM9 e_LUMO partition explorer', fillable=False,
)

def server(input, output, session):
    completed = reactive.value(None)
    message = reactive.value('Enter a SMILES string and click Analyze molecule.')
    busy = reactive.value(False)

    @ui.bind_task_button(button_id='analyze')
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
        if busy():
            return
        try:
            ks = tuple(int(k) for k in input.ks())
            iterations, seed = input.iterations(),input.seed()
            beta, weight, singleton = input.beta(),input.weight(),input.singleton()
            if not input.smiles().strip() or not ks:
                raise ValueError('Enter a SMILES string and select at least one k.')
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
                    int(seed),float(weight),float(singleton),MODEL_DIR)
            completed.set(None)
            message.set('Running Metropolis–Hastings… Results will appear when all selected k values finish.')
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
            message.set('Analysis complete. Results correspond to the submitted settings.')

    @render.ui
    def status():
        return ui.div(message(),class_='alert alert-info',role='status')

    @render.ui
    def results():
        payload = completed()
        if payload is None:
            return ui.p('Saved model files required: buhito_featurizer.pkl and buhito_hierarchical_model_e_lumo.pkl.')
        panels = [ui.nav_panel(f"k = {r['k']}",ui.output_plot(f'plot_{r["k"]}',height='760px'),
                              ui.download_button(f'png_{r["k"]}','Download PNG'))
                  for r in payload['results']]
        return ui.div(
            ui.h4(f"Predicted e_LUMO: {payload['prediction']:.6g} (dataset units)"),
            ui.p('SMILES: '+payload['smiles']),
            ui.navset_tab(*panels),
            ui.h4('Result summary'),ui.output_data_frame('summary'),
            ui.download_button('csv_download','Download summary CSV'),
            ui.download_button('atom_download','Download atom partitions CSV'))

    # Register fixed output IDs once; only selected tabs are displayed.
    def register_k(k):
        @render.plot
        def plot():
            payload = completed()
            req(payload is not None)
            indexes = [i for i,r in enumerate(payload['results']) if r['k']==k]
            req(bool(indexes))
            return result_figure(payload,indexes[0])
        output(id=f'plot_{k}')(plot)

        @render.download(filename=lambda:f'e_lumo_mh_k{k}.png')
        def png():
            payload = completed()
            req(payload is not None)
            index = next(i for i,r in enumerate(payload['results']) if r['k']==k)
            buf = io.BytesIO()
            result_figure(payload,index).savefig(buf,format='png',dpi=180)
            yield buf.getvalue()
        output(id=f'png_{k}')(png)

    for k in range(2,6): register_k(k)

    def rows(payload):
        return [dict(SMILES=payload['smiles'],k=r['k'],Prediction=r['prediction'],
            Raw=r['raw'],Penalty=r['penalty'],Adjusted=r['adjusted'],
            Within=r['signed'][0],Between=r['signed'][1],Higher=r['signed'][2],
            Reconstruction_error=r['error'],**payload['settings']) for r in payload['results']]

    @render.data_frame
    def summary():
        import pandas as pd
        req(completed() is not None)
        return render.DataGrid(pd.DataFrame(rows(completed())),height='250px')

    @render.download(filename='e_lumo_mh_summary.csv')
    def csv_download():
        req(completed() is not None)
        data=rows(completed()); buf=io.StringIO()
        writer=csv.DictWriter(buf,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
        yield buf.getvalue().encode()

    @render.download(filename='e_lumo_mh_atom_partitions.csv')
    def atom_download():
        from rdkit import Chem
        payload=completed();req(payload is not None)
        mol=Chem.Mol(payload['mol']);buf=io.StringIO();writer=csv.writer(buf)
        writer.writerow(['SMILES','k','atom_index','element','partition'])
        for r in payload['results']:
            for a,p in sorted(r['partition'].items()):
                writer.writerow([payload['smiles'],r['k'],a,mol.GetAtomWithIdx(a).GetSymbol(),p])
        yield buf.getvalue().encode()

app = App(app_ui, server)
app.on_shutdown(lambda: POOL.shutdown(wait=False,cancel_futures=True))
