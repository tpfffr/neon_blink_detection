import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent

sys.path.append(str(base_dir))
sys.path.append(str(base_dir / "src"))
sys.path.append(str(base_dir / "functions"))

from functions.pipeline import get_classifier_params
from src.blink_detector import get_params
from training.dataset_splitter import load_dataset_splitter
from training.helper import (
    get_experiment_name_new,
    get_export_dir,
    get_of_params_options,
    get_test_recording_ids,
    get_training_dir,
)
from training.submit import submit_one_job


def main(n_splits=5):
    dataset_splitter = load_dataset_splitter(n_clips=None, n_splits=n_splits)
    clip_names_test = ["2023-01-27_15-59-54-49a115d5"]

    use_pretrained_classifier = False
    use_cluster = False

    classifier_params = get_classifier_params()
    of_params, pp_params = get_params()
    of_params_options = get_of_params_options()

    training_dir = get_training_dir(classifier_params.name, use_pretrained_classifier)
    export_dir = get_export_dir(classifier_params.name, use_pretrained_classifier)
    for of_params in of_params_options:
        experiment_name = get_experiment_name_new(of_params)
        save_path = training_dir / experiment_name
        export_path = export_dir / experiment_name

        # if is_trained(experiment_name, n_splits, export_dir):
        #     print(f"{experiment_name} was done.")
        #     continue
        print(f"save_path={save_path}")

        submit_one_job(
            dataset_splitter,
            clip_names_test,
            classifier_params,
            of_params,
            pp_params,
            export_path=export_path,
            save_path=save_path,
            use_pretrained_classifier=use_pretrained_classifier,
            use_cluster=use_cluster,
        )


if __name__ == "__main__":
    main()
