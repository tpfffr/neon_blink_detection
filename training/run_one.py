import logging
import time
import typing as T
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata
from functions.classifiers import Classifier, load_predictions, save_predictions
from functions.pipeline import post_process_debug
from functions.utils import print_run_time
from src.event_array import Samples
from src.helper import OfParams, PPParams, AugParams
from src.metrics import Scores, ScoresList
from training.dataset_splitter import DatasetSplitter
from video_loader import video_loader
from src.post_processing import classify
from training.datasets_loader import (
    # Datasets,
    concatenate,
    concatenate_all_samples,
    load_samples,
    save_samples,
)
from src.features_calculator import create_grids
from training.evaluation import evaluate
from training.helper import ClassifierParams, Results
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from training.cnn import OpticalFlowCNN, OpticalFlowDataset
import torch
from sklearn.model_selection import train_test_split

logger = logging.getLogger("main")
from sklearn.model_selection import KFold
from copy import copy
from functions.pipeline import get_classifier_params


def main(
    dataset_splitter: DatasetSplitter,
    clip_names_test: T.List[str],
    classifier_params: ClassifierParams,
    of_params: OfParams,
    pp_params: PPParams,
    aug_params: AugParams,
    export_path: Path,
    save_path: Path,
    use_pretrained_classifier: bool,
):
    assert isinstance(dataset_splitter, DatasetSplitter)
    assert isinstance(classifier_params, ClassifierParams)
    assert isinstance(of_params, OfParams)
    assert isinstance(pp_params, PPParams)

    set_logger(save_path)
    logger.info(classifier_params)
    logger.info(of_params)
    logger.info(pp_params)

    t1 = time.perf_counter()

    metrics_sample_train = ScoresList()
    metrics_pp_train = ScoresList()
    metrics_ml_train = ScoresList()

    metrics_sample_val = ScoresList()
    metrics_pp_val = ScoresList()
    metrics_ml_val = ScoresList()

    metrics_sample_test = ScoresList()
    metrics_pp_test = ScoresList()
    metrics_ml_test = ScoresList()

    aug_features = None
    aug_samples = None

    for idx, (clip_names_train, clip_names_val) in enumerate(dataset_splitter):
        (
            all_samples,
            predictions,
            clf_scores,
            aug_features,
            aug_samples,
        ) = collect_samples_and_predict(
            clip_names_train,
            clip_names_val,
            clip_names_test,
            classifier_params,
            of_params,
            aug_params,
            pp_params,
            export_path,
            idx,
            use_pretrained_classifier,
            aug_features,
            aug_samples,
        )

        logger.info("Evaluate full training data")
        metrics_sample, metrics_ml, metrics_pp, n_samples = evaluate_clips(
            clip_names_train, all_samples, predictions, pp_params
        )
        metrics_sample_train.append(metrics_sample, n_samples)
        metrics_ml_train.append(metrics_ml, n_samples)
        metrics_pp_train.append(metrics_pp, n_samples)

        logger.info("Evaluate validation data")
        metrics_sample, metrics_ml, metrics_pp, n_samples = evaluate_clips(
            clip_names_val, all_samples, predictions, pp_params
        )
        metrics_sample_val.append(metrics_sample, n_samples)
        metrics_ml_val.append(metrics_ml, n_samples)
        metrics_pp_val.append(metrics_pp, n_samples)

        logger.info("Evaluate test data")
        metrics_sample, metrics_ml, metrics_pp, n_samples = evaluate_clips(
            clip_names_test, all_samples, predictions, pp_params
        )
        metrics_sample_test.append(metrics_sample, n_samples)
        metrics_ml_test.append(metrics_ml, n_samples)
        metrics_pp_test.append(metrics_pp, n_samples)

    t2 = time.perf_counter()

    results = Results(
        save_path.name,
        classifier_params,
        dataset_splitter,
        of_params,
        pp_params,
        t2 - t1,
        metrics_sample_train,
        metrics_ml_train,
        metrics_pp_train,
        metrics_sample_val,
        metrics_ml_val,
        metrics_pp_val,
        metrics_sample_test,
        metrics_ml_test,
        metrics_pp_test,
        clf_scores,
    )
    results.dump(save_path)

    print_run_time("exp", t2 - t1)


def collect_samples_and_predict(
    clip_names_train: T.List[str],
    clip_names_val: T.List[str],
    clip_names_test: T.List[str],
    classifier_params: ClassifierParams,
    of_params: OfParams,
    aug_options: AugParams,
    pp_params: PPParams,
    export_path: Path,
    idx: int,
    use_pretrained_classifier: bool,
    augmented_features: T.Optional[T.Dict[str, np.ndarray]] = None,
    augmented_samples: T.Optional[T.Dict[str, np.ndarray]] = None,
):
    if not use_pretrained_classifier:

        if idx == 0:
            datasets = video_loader(of_params, aug_options)
        else:
            datasets = video_loader(
                of_params, aug_options, augmented_features, augmented_samples
            )

        augment_data = False

        logger.info("Collect subsampled training data")
        datasets.collect(clip_names_train, bg_ratio=1, augment=augment_data, idx=idx)

        if augment_data:
            augmented_features = datasets.all_aug_features
            augmented_samples = datasets.all_aug_samples
            n_aug_features = sum(
                [augmented_features[x].shape[0] for x in clip_names_train]
            )
        else:
            n_aug_features = 0

        logger.info("augmented features = %d", n_aug_features)

        logger.info("Start training")
        # classifier, xgb_classifier, scores = train_cnn(
        #     datasets,
        #     clip_names_train,
        #     clip_names_val,
        #     export_path,
        #     idx,
        #     augment_data=augment_data,
        #     pp_params=pp_params,
        #     of_params=of_params,
        # )

        classifier, xgb_classifier, scores = train_classifier(
            datasets,
            clip_names_train,
            classifier_params,
            export_path,
            idx,
            augment_data=augment_data,
            pp_params=pp_params,
        )

        logger.info("Collect all training data")
        datasets.collect(clip_names_train)

        logger.info("Collect validation data")
        datasets.collect(clip_names_val)

        logger.info("Collect test data")
        datasets.collect(clip_names_test)

        all_samples = datasets.all_samples
        save_samples(all_samples, export_path, idx)

        logger.info("Predict training & validation & test data")

        predictions = classifier.predict_all_clips(datasets.all_features)

        # if second classifier is used, predict with xgb classifier
        # --------------------------------------------
        # if second_classifier:
        # xgb_predictions = {}

        # for clip_tuple, features in predictions.items():

        #     cnn_features = get_feature_indices(length=50, features=features)
        #     xgb_predictions[clip_tuple] = xgb_classifier.predict(cnn_features)

        # predictions = xgb_predictions
        # --------------------------------------------

        save_predictions(export_path, idx, predictions)
    else:
        logger.info("Load training & validation data")
        all_samples = load_samples(export_path, idx)
        predictions = load_predictions(export_path, idx)

    return all_samples, predictions, scores, augmented_features, augmented_samples


def train_classifier(
    datasets: video_loader,
    clip_names: T.List[str],
    classifier_params: ClassifierParams,
    export_path: Path,
    idx: int,
    augment_data: bool,
    pp_params: PPParams,
):
    features = concatenate(datasets.all_features, clip_names)
    samples_gt = concatenate_all_samples(datasets.all_samples, clip_names)
    labels = samples_gt.labels

    if augment_data:
        aug_features = concatenate(datasets.all_aug_features, clip_names)
        aug_samples_gt = concatenate_all_samples(datasets.all_aug_samples, clip_names)
        aug_labels = aug_samples_gt.labels

        features = np.concatenate((features, aug_features), axis=0)
        labels = np.concatenate((labels, aug_labels), axis=0)

    classifier = Classifier(classifier_params, export_path)
    classifier.on_fit(features, labels)

    predictions = classifier.predict(features)

    # cnn_features = get_feature_indices(length=50, features=predictions)

    # classifier_params = get_classifier_params()
    # xgb_classifier = Classifier(classifier_params)
    # xgb_classifier.on_fit(features=cnn_features, labels=labels)
    # predictions = xgb_classifier.predict(cnn_features)
    xgb_classifier = None

    predictions = classify(predictions, pp_params)

    clf_scores = compute_clf_scores(predictions, labels)

    logger.info("Classifier scores:")
    logger.info(
        f"Sample-based recall on = {clf_scores['recall_on']:.2f}, "
        f"precision on = {clf_scores['precision_on']:.2f}, "
        f"F1 on = {clf_scores['f1_on']:.2f}"
    )
    logger.info(
        f"Sample-based recall off = {clf_scores['recall_off']:.2f}, "
        f"precision off = {clf_scores['precision_off']:.2f}, "
        f"F1 off = {clf_scores['f1_off']:.2f}"
    )
    logger.info(
        f"Sample-based recall bg = {clf_scores['recall_bg']:.2f}, "
        f"precision bg = {clf_scores['precision_bg']:.2f}, "
        f"F1 bg = {clf_scores['f1_bg']:.2f}"
    )

    classifier.save_base_classifier(idx)

    return classifier, xgb_classifier, clf_scores


def train_cnn(
    datasets: video_loader,
    clip_names: T.List[str],
    clip_names_val: T.List[str],
    export_path: Path,
    idx: int,
    augment_data: bool,
    pp_params: PPParams,
    of_params: OfParams,
):
    """Train a CNN classifier.

    Args:
        datasets: The datasets to train on.
        clip_names: The names of the clips to train on.
        classifier_params: The parameters for the classifier.
        export_path: The path to export the classifier to.
        idx: The index of the current iteration.
        augment_data: Whether to augment the data.
        pp_params: The parameters for post-processing.

        Returns:
            The trained classifier and the scores.
    """

    # Concatenate all features and labels
    features = concatenate(datasets.all_features, clip_names)

    # reshape features for CNN
    n_layers = of_params.n_layers
    grid_size = of_params.grid_size

    features_l = features[:, : (features.shape[1] // 2)]
    features_r = features[:, (features.shape[1] // 2) :]

    features_l = features_l.reshape(-1, n_layers, grid_size, grid_size)
    features_r = features_r.reshape(-1, n_layers, grid_size, grid_size)

    features = np.concatenate([features_l, features_r], axis=1)

    samples_gt = concatenate_all_samples(datasets.all_samples, clip_names)
    labels = samples_gt.labels

    # # Augment data
    # if augment_data:
    #     aug_features = concatenate(datasets.all_aug_features, clip_names)
    #     aug_samples_gt = concatenate_all_samples(datasets.all_aug_samples, clip_names)
    #     aug_labels = aug_samples_gt.labels

    #     features = np.concatenate((features, aug_features), axis=0)
    #     labels = np.concatenate((labels, aug_labels), axis=0)

    # Split into train and validation set for early stopping
    # Stratify to ensure that the classes are balanced in both sets
    # if idx == 0:
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, stratify=labels, test_size=0.075, random_state=42
    )
    # else:
    #     X_train = features
    #     y_train = labels
    #     X_val = concatenate(datasets.all_features, clip_names_val)
    #     samples_gt = concatenate_all_samples(datasets.all_samples, clip_names_val)
    #     y_val = samples_gt.labels

    # Convert to torch tensors
    X_train = torch.from_numpy(X_train).float().cuda()
    y_train = torch.from_numpy(y_train).long().cuda()
    X_val = torch.from_numpy(X_val).float().cuda()
    y_val = torch.from_numpy(y_val).long().cuda()

    # Create classifier
    classifier = OpticalFlowCNN(export_path, of_params)
    classifier.cuda()

    # Train classifier
    classifier.training_func(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=32,
        num_epochs=100,
    )

    predictions = classifier.predict(X_train)
    predictions_discrete = classify(predictions, pp_params)
    clf_scores = compute_clf_scores(predictions_discrete, y_train.cpu().numpy())

    logger.info("Classifier scores (RAW):")
    logger.info(
        f"Sample-based recall on = {clf_scores['recall_on']:.2f}, "
        f"precision on = {clf_scores['precision_on']:.2f}, "
        f"F1 on = {clf_scores['f1_on']:.2f}"
    )
    logger.info(
        f"Sample-based recall off = {clf_scores['recall_off']:.2f}, "
        f"precision off = {clf_scores['precision_off']:.2f}, "
        f"F1 off = {clf_scores['f1_off']:.2f}"
    )
    logger.info(
        f"Sample-based recall bg = {clf_scores['recall_bg']:.2f}, "
        f"precision bg = {clf_scores['precision_bg']:.2f}, "
        f"F1 bg = {clf_scores['f1_bg']:.2f}"
    )

    cnn_features = get_feature_indices(length=50, features=predictions)

    classifier_params = get_classifier_params()
    xgb_classifier = Classifier(classifier_params)
    xgb_classifier.on_fit(features=cnn_features, labels=y_train.cpu())
    predictions = xgb_classifier.predict(cnn_features)

    predictions = classify(predictions, pp_params)
    clf_scores = compute_clf_scores(predictions, y_train.cpu().numpy())

    logger.info("Classifier scores (AFTER SECOND CLASSIFIER):")
    logger.info(
        f"Sample-based recall on = {clf_scores['recall_on']:.2f}, "
        f"precision on = {clf_scores['precision_on']:.2f}, "
        f"F1 on = {clf_scores['f1_on']:.2f}"
    )
    logger.info(
        f"Sample-based recall off = {clf_scores['recall_off']:.2f}, "
        f"precision off = {clf_scores['precision_off']:.2f}, "
        f"F1 off = {clf_scores['f1_off']:.2f}"
    )
    logger.info(
        f"Sample-based recall bg = {clf_scores['recall_bg']:.2f}, "
        f"precision bg = {clf_scores['precision_bg']:.2f}, "
        f"F1 bg = {clf_scores['f1_bg']:.2f}"
    )

    # Save classifier
    classifier.save_base_classifier(idx)

    return classifier, xgb_classifier, clf_scores


def get_feature_indices(length: int, features: np.ndarray, indices: np.ndarray = None):
    if indices is None:
        indices = np.arange(0, features.shape[0])

    all_indices = np.array(
        [np.arange(index - length, index + length) for index in indices]
    )
    all_indices = np.clip(all_indices, 0, max(indices) - 1)

    return np.array(features[all_indices, :].reshape(-1, 2 * length * 3))


def compute_clf_scores(predictions, labels):

    scores = {}

    recall = recall_score(labels, predictions, average=None)
    scores.update(
        {"recall_bg": recall[0], "recall_on": recall[1], "recall_off": recall[2]}
    )

    precision = precision_score(labels, predictions, average=None)
    scores.update(
        {
            "precision_bg": precision[0],
            "precision_on": precision[1],
            "precision_off": precision[2],
        }
    )

    f1 = f1_score(labels, predictions, average=None)
    scores.update({"f1_bg": f1[0], "f1_on": f1[1], "f1_off": f1[2]})

    return scores


def evaluate_clips(
    clip_names: T.List[str],
    all_samples: T.Dict[str, Samples],
    predictions: T.Dict[str, np.ndarray],
    pp_params: PPParams,
) -> T.Tuple[Scores, Scores, Scores, int]:
    samples_gt = concatenate_all_samples(all_samples, clip_names)
    proba = concatenate(predictions, clip_names)

    blink_array_pd_pp, samples_pd = post_process_debug(
        samples_gt.timestamps, proba, pp_params
    )
    metrics_sample, metrics_ml, metrics_pp = evaluate(
        samples_gt, samples_pd, blink_array_pd_pp
    )
    n_samples = len(samples_gt)
    return metrics_sample, metrics_ml, metrics_pp, n_samples


def set_logger(save_path: Path):
    fmt = "%(asctime)s [%(levelname)s]\t%(message)s"
    datefmt = "%Y/%m/%d %H:%M:%S"
    log_path = save_path / "debug.log"
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(logging.Formatter(fmt, datefmt))
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
