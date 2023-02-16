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

logger = logging.getLogger("main")
from sklearn.model_selection import KFold


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

    for idx, (clip_names_train, clip_names_val) in enumerate(dataset_splitter):
        all_samples, predictions = collect_samples_and_predict(
            clip_names_train,
            clip_names_val,
            clip_names_test,
            classifier_params,
            of_params,
            aug_params,
            export_path,
            idx,
            use_pretrained_classifier,
        )

        logger.info("Evaluate training data")
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
    )
    results.dump(save_path)

    print_run_time("exp", t2 - t1)


def collect_samples_and_predict(
    clip_names_train: T.List[str],
    clip_names_val: T.List[str],
    clip_names_test: T.List[str],
    classifier_params: ClassifierParams,
    of_params: OfParams,
    aug_params: AugParams,
    export_path: Path,
    idx: int,
    use_pretrained_classifier: bool,
):
    if not use_pretrained_classifier:
        logger.info("Collect training data")
        datasets = video_loader(of_params, aug_params)

        # add information about dataset to be loaded here
        datasets.collect(clip_names_train, bg_ratio=2, augment=True)

        if datasets.augment:
            n_augmented_features = sum(
                [datasets.augmented_features[x].shape[0] for x in clip_names_train]
            )
        else:
            n_augmented_features = 0

        logger.info("augmented features = %d", n_augmented_features)

        logger.info("Start training")
        classifier = train_classifier(
            datasets, clip_names_train, classifier_params, export_path, idx
        )

        logger.info("Collect validation data")
        datasets.collect(clip_names_val)

        logger.info("Collect test data")
        datasets.collect(clip_names_test)

        all_samples = datasets.all_samples
        save_samples(all_samples, export_path, idx)

        logger.info("Predict training & validation & test data")
        predictions = classifier.predict_all_clips(datasets.all_features)
        save_predictions(export_path, idx, predictions)
    else:
        logger.info("Load training & validation data")
        all_samples = load_samples(export_path, idx)
        predictions = load_predictions(export_path, idx)

    return all_samples, predictions


def train_classifier(
    datasets: video_loader,
    clip_names: T.List[str],
    classifier_params: ClassifierParams,
    export_path: Path,
    idx: int,
):
    features = concatenate(datasets.all_features, clip_names)
    samples_gt = concatenate_all_samples(datasets.all_samples, clip_names)
    labels = samples_gt.labels

    grid_size = 20
    of_grid = create_grids(datasets._of_params.img_shape, grid_size, full_grid=True)

    small_grid = create_grids(
        datasets._of_params.img_shape,
        datasets._of_params.grid_size + 2,
        full_grid=False,
    )
    n_rep = datasets._of_params.n_layers * 2

    small_grid = np.concatenate(n_rep * [small_grid])
    of_grid = np.concatenate(n_rep * [of_grid])

    n_grid_points = datasets._of_params.grid_size**2

    features = griddata(of_grid, features.transpose(), small_grid, method="nearest")

    features = features.transpose()

    if datasets.augment:
        augmented_features = concatenate(datasets.augmented_features, clip_names)
        augmented_samples_gt = concatenate_all_samples(
            datasets.augmented_samples, clip_names
        )

        features = np.concatenate([features, augmented_features])
        labels = np.concatenate([labels, augmented_samples_gt.labels])

    classifier = Classifier(classifier_params, export_path)
    classifier.on_fit(features, labels)
    classifier.save_base_classifier(idx)
    return classifier


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
