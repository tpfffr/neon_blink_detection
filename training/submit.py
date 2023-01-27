import pickle
import shutil
import typing as T
from pathlib import Path

import numpy as np
from plml.cluster.submit_job import submit_job_to_gridengine

from src.helper import OfParams, PPParams
from training.dataset_splitter import DatasetSplitter
from training.helper import ClassifierParams


def submit_one_job(
    dataset_splitter: DatasetSplitter,
    clip_names_test: T.List[str],
    classifier_params: ClassifierParams,
    of_params: OfParams,
    pp_params: PPParams,
    export_path: Path,
    save_path: Path,
    use_pretrained_classifier: bool,
    use_cluster: bool,
):
    # shutil.rmtree(export_path, True)
    export_path.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(save_path, True)

    kwargs = {
        "dataset_splitter": dataset_splitter,
        "clip_names_test": clip_names_test,
        "classifier_params": classifier_params,
        "of_params": of_params,
        "pp_params": pp_params,
        "export_path": export_path,
        "save_path": save_path,
        "use_pretrained_classifier": use_pretrained_classifier,
    }
    experiment_name = save_path.name
    print(f"experiment_name={experiment_name}")

    if use_cluster:
        if use_pretrained_classifier:
            ram_gb = 0.5
        else:
            ram_gb = (
                80000
                * dataset_splitter.n_clips
                * of_params.n_layers
                * of_params.grid_size**2
                / 1e9
            )
            print(f"estimate ram_gb={ram_gb}")

        submit_job_to_gridengine(
            save_path=str(save_path),
            environment="tom_py310",
            script_path=Path(__file__).with_name("run_one.py"),
            fn_name="main",
            kwargs=kwargs,
            ram_gb=np.ceil(ram_gb),
            gpus=0,
            job_name=experiment_name,
            queue_name="cpu.q",
            reproducible=False,
        )
    else:
        from plml.cluster.submit_job import _save_kwargs

        import run_one

        save_path.mkdir(parents=True, exist_ok=True)
        _save_kwargs(kwargs, save_path)
        with open(save_path / "kwargs.pkl", "rb") as file:
            kwargs = pickle.load(file)
        run_one.main(**kwargs)
