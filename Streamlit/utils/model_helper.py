import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from utils.custom_transformers import P95Capper
from utils.constants import FEATURE_ORDER, MODEL_PATH

sys.modules["__main__"].P95Capper = P95Capper

if "__mp_main__" not in sys.modules:
    sys.modules["__mp_main__"] = sys.modules["__main__"]
sys.modules["__mp_main__"].P95Capper = P95Capper


@st.cache_resource
def load_model():
    import types

    if "utils.custom_transformers" not in sys.modules:
        fake_mod = types.ModuleType("utils.custom_transformers")
        fake_mod.P95Capper = P95Capper
        sys.modules["utils.custom_transformers"] = fake_mod

    obj = joblib.load(MODEL_PATH)
    pipeline = obj["model"]

    preprocessor = pipeline.named_steps['preprocessor']
    for _, transformer, _ in preprocessor.transformers:
        if hasattr(transformer, 'named_steps'):
            for step_name, step in transformer.named_steps.items():
                if type(step).__name__ == 'P95Capper':
                    caps = step.__dict__.get('caps_')
                    step.__class__ = P95Capper
                    if caps is not None:
                        step.caps_ = caps

    return pipeline, float(obj["best_score"])


def predict(df, pipeline):
    df = df[FEATURE_ORDER].copy()
    probas = pipeline.predict_proba(df)[:, 1]
    preds  = (probas >= 0.5).astype(int)
    return preds, probas
