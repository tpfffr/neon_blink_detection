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

warnings.filterwarnings("default")

path = "/users/tom/experiments/neon_blink_detection/datasets/train_data/optical_flow/all_timestamps.pkl"

with open(path, "rb") as f:
    all_timestamps = pickle.load(f)


def compute_confidence(type="mean", prctile=None):

    tp_confidence = {}
    fp_confidence = {}

    blink_info = {}

    dataset_splitter = load_dataset_splitter(n_clips=None, n_splits=5)
    for idx, (_, clip_tuples_val) in enumerate(dataset_splitter):

        print(idx)

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

            print(clip_name)

            tp_confidence[clip_name] = []
            fp_confidence[clip_name] = []

            pred_blink, gt_blinks, _, _, _ = pf.get_blink_events(
                clip_name, clf, all_probas[clip_name], all_timestamps
            )

            _, pp_params = pf.get_params()

            smoothed_proba = smooth_proba(all_probas[clip_name], pp_params)
            _, tp, fn, fp = pf.compute_multiple_iou(gt_blinks, pred_blink)

            fpr, br, blink_nr = pf.compute_fpr_and_br(
                clip_name, fp, tp, fn, all_timestamps
            )

            blink_info[clip_name] = {}
            blink_info[clip_name]["tp"] = tp
            blink_info[clip_name]["fp"] = fp
            blink_info[clip_name]["fn"] = fn
            blink_info[clip_name]["br"] = br
            blink_info[clip_name]["fpr"] = fpr
            blink_info[clip_name]["blink_nr"] = blink_nr
            blink_info[clip_name]["gt"] = gt_blinks
            blink_info[clip_name]["smoothed_proba"] = smoothed_proba

            for i in range(len(tp)):
                start_idx = tp[i][1][0]
                end_idx = tp[i][1][1]

                confidence_blink_tmp, _, _ = pf.compute_confidence(
                    start_idx, end_idx, smoothed_proba, type=type, prctile=prctile
                )
                print(confidence_blink_tmp)

                tp_confidence[clip_name].append(confidence_blink_tmp)

            for i in range(len(fp)):
                start_idx = fp[i][1][0]
                end_idx = fp[i][1][1]

                confidence_blink_tmp, _, _ = pf.compute_confidence(
                    start_idx, end_idx, smoothed_proba, type=type, prctile=prctile
                )
                fp_confidence[clip_name].append(confidence_blink_tmp)

    return tp_confidence, fp_confidence, blink_info


dataset_splitter = load_dataset_splitter(n_clips=None, n_splits=5)
tp_confidence, fp_confidence, blink_info = compute_confidence(type="mean")
