import functools
import logging
import random
import subprocess
import time

import cv2
import numpy as np

logger = logging.getLogger("main")


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        t1 = time.perf_counter()
        output = func(*args, **kwargs)
        t2 = time.perf_counter()

        run_time = t2 - t1
        print_run_time(func.__name__, run_time)

        return output

    return wrapper_timer


def print_run_time(func_name, run_time_s):
    show_time = get_show_time(run_time_s)

    if logger.handlers:
        logger.debug(f"{func_name} took {show_time}")
    else:
        print(f"{func_name} took {show_time}")

    return f"{func_name} took {show_time}"


def get_show_time(run_time_s):
    if run_time_s < 1e-3:
        show_time = f"{run_time_s * 1e6:.0f} mus"
    elif run_time_s < 1:
        show_time = f"{run_time_s * 1e3:.0f} ms"
    elif run_time_s < 60:
        show_time = f"{run_time_s:.1f} s"
    elif run_time_s >= 3600:
        hours = run_time_s // 3600
        minutes = (run_time_s - hours * 3600) // 60
        seconds = run_time_s - hours * 3600 - minutes * 60
        show_time = f"{hours:.0f} h {minutes:.0f} m {seconds:.0f} s"
    else:
        minutes = run_time_s // 60
        seconds = run_time_s - minutes * 60
        show_time = f"{minutes:.0f} m {seconds:.0f} s"

    return show_time


def random_sample(population, k=None):
    random.seed(0)
    k = len(population) if k is None else min(k, len(population))
    samples = random.sample(population, k)
    return samples


def get_sns_colors(n_colors=8, palette="Set1", bgr=True):
    import seaborn as sns

    colors = sns.color_palette(palette, n_colors)
    colors = np.asarray(colors, dtype=np.float32).reshape(-1, 1, 3)
    if bgr:
        colors = cv2.cvtColor(colors, cv2.COLOR_RGB2BGR)
    colors *= 255
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    return colors.tolist()


def set_logger(save_path):
    fmt = "%(asctime)s [%(levelname)s]\t%(message)s"
    datefmt = "%Y/%m/%d %H:%M:%S"
    log_path = save_path / "debug.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt))
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)


def execute(arg_str, log_path="", get_stdout=False):
    popenargs = arg_str.split(" ")
    if log_path:
        get_stdout = True
    stdout = subprocess.PIPE if get_stdout else subprocess.DEVNULL

    start_time = time.perf_counter()
    res = subprocess.run(popenargs, check=True, universal_newlines=True, stdout=stdout)
    end_time = time.perf_counter()

    if log_path:
        with open(log_path, "a") as f:
            f.write(res.stdout)

    run_time = end_time - start_time
    return run_time, res.stdout


def get_eye_timestamps(rec):
    return np.concatenate([part.times.values for part in rec.eye_left.parts])
