import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_registered_fit_count_and_non_gpu_scope():
    cfg = json.loads((ROOT/'configs/experiments/p3_numeric_cpudet_s3_20260907_v1.json').read_text())
    assert (5+1)*len(cfg['seeds'])*2 == cfg['backbone_fits_per_cold'] == 36
    assert cfg['router_fits_per_cold'] == 5
    assert cfg['cold_repetitions'] == 2
    assert cfg['cpu_threads'] == 4 and cfg['gpu_workers'] == 0
    assert cfg['uploads'] == cfg['deletions'] == cfg['final_submission'] == 0


def test_component_ensemble_clips_each_seed_before_average():
    tree = ast.parse((ROOT/'scripts/p3_cpudet_final_day_20260907_v1.py').read_text())
    node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'SeedEnsemble')
    scope = {'np':np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<isolated-ensemble>','exec'),scope)

    class Constant:
        def __init__(self, value):
            self.value = value

        def predict(self, frame, thread_count):
            assert thread_count == 4
            return np.full(len(frame), self.value)

    frame=pd.DataFrame({'current_hs_for_residual':[2.,2.]})
    ensemble=scope['SeedEnsemble']([Constant(-10),Constant(2)],False)
    np.testing.assert_array_equal(ensemble.predict(frame)+2,[2,2])


def test_cpu_parameters_override_parent_gpu():
    tree=ast.parse((ROOT/'scripts/p3_cpudet_final_day_20260907_v1.py').read_text())
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='cpu_parameters')
    scope={'ORIGINAL_PARAMETERS':lambda *args,**kwargs: {'task_type':'GPU','devices':'0','thread_count':2}}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<cpu-contract>','exec'),scope)
    assert scope['cpu_parameters']({},'multi',1)=={'task_type':'CPU','thread_count':4}
