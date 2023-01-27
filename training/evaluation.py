import typing as T

from src.event_array import EventArray, MatchedEventArray, Samples
from src.label_mapper import label_mapping
from src.metrics import Scores, calculate_basic_scores


def evaluate(
    samples_gt: Samples, samples_pd: Samples, blink_array_pd_pp: EventArray
) -> T.Tuple[Scores, Scores, Scores]:
    metrics_sample = get_sample_based_metrics(samples_gt, samples_pd)
    metrics_ml = get_event_based_metrics(samples_gt.blink_array, samples_pd.blink_array)
    metrics_pp = get_event_based_metrics(samples_gt.blink_array, blink_array_pd_pp)
    return metrics_sample, metrics_ml, metrics_pp


def get_sample_based_metrics(samples_gt: Samples, samples_pd: Samples) -> Scores:
    label_blink = label_mapping.blink
    label_bg = label_mapping.bg

    blink_labels_gt = samples_gt.blink_labels
    blink_labels_pd = samples_pd.blink_labels

    metrics_sample = calculate_basic_scores(
        blink_labels_gt, blink_labels_pd, label_blink, label_bg
    )
    # print(f"Sample-based metrics: {metrics_sample}")
    return metrics_sample


def get_event_based_metrics(
    blink_array_gt: EventArray, blink_array_pd: EventArray, iou_thr: float = 0.2
) -> Scores:
    label_blink = label_mapping.blink
    label_bg = label_mapping.bg

    metrics_event = MatchedEventArray(
        blink_array_gt, blink_array_pd, label_blink, label_bg, iou_thr
    ).get_scores()
    # print(f"Event-based metrics: {metrics_event}")
    return metrics_event
