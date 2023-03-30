import sys
from pathlib import Path
import numpy as np

sys.path.append("/users/tom/git/neon_blink_detection/")


from training.helper import OfParams, PPParams, AugParams
from training.video_loader import video_loader
import notebooks.plotting_functions as pf
from training.dataset_splitter import load_dataset_splitter

# rec
aug_params = AugParams()
of_params, pp_params = pf.get_params()

dataset_splitter = load_dataset_splitter(n_clips=None, n_splits=5)
for idx, (_, clip_tuples_val) in enumerate(dataset_splitter):

    if idx == 0:
        continue

    try:

        rec = video_loader(of_params, aug_params)

        for clip_name in clip_tuples_val:

            ts, left_images, right_images = rec._get_frames(
                clip_name, convert_to_gray=True
            )

            # save timestamps - make dir if it doesn't exist
            Path(
                "/users/tom/experiments/neon_blink_detection/datasets/train_data/optical_flow/clip_timestamps/"
            ).mkdir(parents=True, exist_ok=True)

            np.save(
                Path(
                    "/users/tom/experiments/neon_blink_detection/datasets/train_data/optical_flow/clip_timestamps/"
                )
                / f"{clip_name}.npy",
                ts,
            )

    except Exception as e:
        print(e)
