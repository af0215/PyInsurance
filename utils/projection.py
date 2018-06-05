__author__ = 'Ting'

import pandas as pd

from Infra.CurveAggregator import create_curve_aggregator
from Managers.ProjectionManager import ProjectionManager


def run_projection(model_iters, metrics, params):
    dfs = []
    for model_iter, metric, param in zip(model_iters, metrics, params):
        crv_aggregator = create_curve_aggregator(metric)
        proj_mgr = ProjectionManager(crv_aggregator, model_iter, **param)
        proj_mgr.run()
        dfs.append( crv_aggregator.to_dataframe())

    return pd.concat(dfs, axis=1)

