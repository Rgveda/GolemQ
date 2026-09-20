# coding:utf-8
"""
Feature review stubs (to be populated).

Functions referenced by services/persistence/ for attaching reality features.
"""
import pandas as pd


def attach_reality_features(features_dummy, annual=1008, collections=None):
    """Attach reality features to a baseline dataframe (stub)."""
    return features_dummy.copy() if features_dummy is not None else pd.DataFrame()
