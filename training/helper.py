import itertools
import logging
import pickle
import pprint
import typing as T
from dataclasses import dataclass, field
from pathlib import Path

from src.helper import OfParams, PPParams, AugParams
from src.metrics import ScoresList
from training.dataset_splitter import DatasetSplitter

from datetime import datetime

logger = logging.getLogger("main")


root_dir = Path("/cluster/users/tom/git/neon_blink_detection")
classifier_name_default = "XGBClassifier-3"


@dataclass
class ClassifierParams:
    """Parameters for selecting the datasets"""

    name: str
    algorithm: T.Any
    kwargs: dict = field(default_factory=dict)

    def __str__(self):
        return f"Classifier: {repr(self)}"


def get_of_params_options():
    n_layers_options = [7]  # used to be [3, 5, 7]
    layer_interval_options = [7]  # used to be [1, 3, 5, 7]
    average_options = [False]
    img_shape_options = [(64, 64)]
    grid_size_options = [4]  # used to be [4, 7, 10]
    step_size_options = [5]
    window_size_options = [15]  # used to be [[7, 11, 15]]
    stop_steps_options = [3]

    options = itertools.product(
        n_layers_options,
        layer_interval_options,
        average_options,
        img_shape_options,
        grid_size_options,
        step_size_options,
        window_size_options,
        stop_steps_options,
    )
    options = list(options)
    of_params_options = sorted(set(OfParams(*option) for option in options))

    print(f"options {len(of_params_options)}")
    return of_params_options


def get_augmentation_options():
    xy_shift = [0.3]
    zoom = [0.4]

    options = itertools.product(
        xy_shift,
        zoom,
    )

    options = list(options)
    aug_params_options = sorted(set(AugParams(*option) for option in options))

    print(f"options {len(aug_params_options)}")
    return aug_params_options


def get_export_dir(classifier_name=None, use_pretrained_classifier=False):
    classifier_name = classifier_name or classifier_name_default
    export_dir = root_dir / f"export-{classifier_name}"

    if use_pretrained_classifier == False:
        while export_dir.is_dir():
            dt = datetime.now().strftime("%d %m %Y %H %M").replace(" ", "")
            export_dir = Path(str(export_dir) + "-%s" % dt)

    return export_dir


def get_training_dir(classifier_name=None, use_pretrained_classifier=False):
    classifier_name = classifier_name or classifier_name_default
    training_dir = root_dir / f"training-{classifier_name}"

    if use_pretrained_classifier == False:
        while training_dir.is_dir():
            dt = datetime.now().strftime("%d %m %Y %H %M").replace(" ", "")
            training_dir = Path(str(training_dir) + "-%s" % dt)

    return training_dir


@dataclass
class Results:
    experiment_name: str
    classifier_params: ClassifierParams
    dataset_splitter: DatasetSplitter
    of_params: OfParams
    pp_params: PPParams
    run_time: float
    metrics_sample_train: ScoresList
    metrics_ml_train: ScoresList
    metrics_pp_train: ScoresList
    metrics_sample_val: ScoresList
    metrics_ml_val: ScoresList
    metrics_pp_val: ScoresList
    metrics_sample_test: ScoresList
    metrics_ml_test: ScoresList
    metrics_pp_test: ScoresList

    def dump(self, save_path):
        path = save_path / "results.pkl"
        logger.info(f"Results are saved to {path}")
        pickle.dump(self, open(path, "wb"))
        path.with_suffix(".txt").write_text(pprint.pformat(self, indent=4))
        self.print()

    def print(self):
        logger.info(
            "Training scores:\n"
            f"Sample-based {self.metrics_sample_train}\n"
            f"Event-based-ml {self.metrics_ml_train}\n"
            f"Event-based-pp {self.metrics_pp_train}"
        )
        logger.info(
            "Validation scores:\n"
            f"Sample-based {self.metrics_sample_val}\n"
            f"Event-based-ml {self.metrics_ml_val}\n"
            f"Event-based-pp {self.metrics_pp_val}"
        )
        logger.info(
            "Test scores:\n"
            f"Sample-based {self.metrics_sample_test}\n"
            f"Event-based-ml {self.metrics_ml_test}\n"
            f"Event-based-pp {self.metrics_pp_test}"
        )


def load_results(results_dir) -> T.List[Results]:
    results_dir = Path(results_dir)

    all_results = []
    for save_path in sorted(results_dir.iterdir()):
        results = load_one_result(save_path)
        if results is not None:
            all_results.append(results)

    print(f"Load from {len(all_results)} results")
    return all_results


def load_one_result(save_path: Path) -> Results:
    path = save_path / "results.pkl"
    try:
        results = pickle.load(open(path, "rb"))
    except Exception as err:
        print(err)
        return
    if not isinstance(results, Results):
        return

    try:
        results.metrics_pp_test
    except AttributeError:
        return

    return results


def load_kwargs():
    training_dir = get_training_dir()

    all_kwargs = {}
    for save_path in sorted(training_dir.iterdir()):
        experiment_name = save_path.name
        kwargs_path = save_path / "kwargs.pkl"
        if kwargs_path.is_file() and is_trained(experiment_name):
            all_kwargs[experiment_name] = pickle.load(open(kwargs_path, "rb"))

    print(f"Load from {len(all_kwargs)} kwargs")
    return all_kwargs


def is_trained(experiment_name: str, n_splits: int = 5, export_dir=None):
    export_dir = export_dir or get_export_dir()
    path = export_dir / experiment_name
    trained = True
    for n in range(n_splits + 1):
        trained &= (path / f"samples-{n}.pkl").is_file()
        trained &= (path / f"proba-{n}.npy").is_file()
        trained &= (path / f"weights-{n}.sav").is_file()
    return trained


def get_experiment_name(of_params: OfParams) -> str:
    return (
        f"n_layers={of_params.n_layers}-"
        f"layer_interval={of_params.layer_interval}-"
        f"average={of_params.average}-"
        f"step_size={of_params.step_size}"
    )


def get_feature_dir_name(of_params: OfParams) -> str:
    return (
        f"grid{of_params.grid_size}-"
        f"step{of_params.step_size}-"
        f"win{of_params.window_size}-"
        f"steps{of_params.stop_steps}"
    )


def get_experiment_name_new(of_params: OfParams, aug_params: AugParams) -> str:
    return (
        # "subtract-"
        f"n_layers{of_params.n_layers}-"
        f"layer_interval{of_params.layer_interval}-"
        f"grid{of_params.grid_size}-"
        f"win{of_params.window_size}"
        # f"shift{aug_params.xy_shift}-"
        # f"zoom{aug_params.zoom}"
    )


def get_feature_dir_name_new(of_params: OfParams) -> str:
    return (
        # f"grid{of_params.grid_size}-"
        f"step{of_params.step_size}-"
        f"win{of_params.window_size}-"
        f"steps{of_params.stop_steps}-"
        f"shape{of_params.img_shape[0]}"
    )


# def get_test_recording_ids() -> T.List[str]:
#     return [
#         "834b1e6c-1952-44a8-a5d3-6a0dfb701d2e",
#         "7cc0960d-f982-4da9-b849-cadd79e05291",
#         "6dc932bf-9cf5-40af-b28c-f19b93817259",
#         "76c6a92d-0870-43ce-bcaf-19fec7866207",
#         "4f2f7c16-8783-43ab-a801-2a251ca6b1dd",
#         "abe6ec68-d0c5-4bf1-b3ec-83523e518b93",
#         "321217b2-0dce-4e3c-9e79-a3faf3734e52",
#         "03c2c28e-a0c6-4592-8704-7687ffaac670",
#     ]
