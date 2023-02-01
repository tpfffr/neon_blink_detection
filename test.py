import pandas as pd
from pathlib import Path

labels_path = Path("/home/tom/blink_data")


def load_blink_labels(labels_path: Path, idx: int):
    return pd.read_json(labels_path / "annotations%d.json" % idx)
