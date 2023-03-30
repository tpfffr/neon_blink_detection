import numpy as np
import sys

sys.path.append("/users/tom/git/neon_blink_detection/")

import notebooks.plotting_functions as pf
from src.post_processing import smooth_proba
import pickle
import matplotlib.pyplot as plt
import joblib


from training.dataset_splitter import load_dataset_splitter
import warnings

warnings.filterwarnings("ignore")

path = "/users/tom/experiments/neon_blink_detection/datasets/train_data/optical_flow/all_timestamps.pkl"

with open(path, "rb") as f:
    all_timestamps = pickle.load(f)

# def compute_confidence(type="mean", prctile=None):

tp_confidence = {}
fp_confidence = {}

blink_info = {}

prctile = 0.25
type = "mean"

dataset_splitter = load_dataset_splitter(n_clips=None, n_splits=5)
for idx, (_, clip_tuples_val) in enumerate(dataset_splitter):

    if idx == 0:
        continue

    clf = (
        "/users/tom/git/neon_blink_detection/export-XGBClassifier-3-100320231148/n_lay5-lay_intv7-grid4-win15-trans0.0-scale0.0/weights-%d.sav"
        % idx
    )

    all_probas = np.load(
        "/users/tom/git/neon_blink_detection/export-XGBClassifier-3-100320231148/n_lay5-lay_intv7-grid4-win15-trans0.0-scale0.0/proba-%d.npy"
        % idx,
        allow_pickle=True,
    )

    for clip_name in clip_tuples_val:

        tp_confidence[clip_name] = []
        fp_confidence[clip_name] = []

        # try:

        pred_blink, gt_blinks, _, _, _ = pf.get_blink_events(
            clip_name, clf, all_probas[clip_name], all_timestamps
        )
