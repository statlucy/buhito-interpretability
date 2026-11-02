"""Spawn-safe entry points: importing this module does not create another worker pool."""
def preload(models_dir):
    import matplotlib
    matplotlib.use('Agg')
    from analysis import preload_models
    return preload_models(models_dir)

def run_analysis(*args):
    from analysis import analyze
    return analyze(*args)
