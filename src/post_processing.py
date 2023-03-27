import typing as T

import numpy as np

from event_array import BlinkEvent, EventArray, Samples
from helper import PPParams
from label_mapper import label_mapping


def post_process(
    timestamps: T.Sequence, proba: np.ndarray, pp_params: PPParams
) -> T.List[BlinkEvent]:
    proba = smooth_proba(proba, pp_params)
    pd_labels = classify(proba, pp_params)
    samples_pd = Samples(timestamps, pd_labels)

    event_array_pd = samples_pd.event_array
    blink_array_pd_pp = filter_wrong_sequence(
        event_array_pd, pp_params.max_gap_duration_s
    )
    blink_array_pd_pp = filter_short_events(
        blink_array_pd_pp, pp_params.short_event_min_len_s, label_mapping.blink
    )
    blink_events = blink_array_pd_pp.blink_events
    return blink_events


# def post_process(
#     timestamps: T.Sequence, proba: np.ndarray, pp_params: PPParams
# ) -> T.Tuple[T.List[BlinkEvent], T.List[BlinkEvent]]:
#     proba = smooth_proba(proba, pp_params)
#     pd_labels = classify(proba, pp_params)
#     samples_pd = Samples(timestamps, pd_labels)

#     event_array_pd = samples_pd.event_array
#     blink_array_pd_pp = filter_wrong_sequence(
#         event_array_pd, pp_params.max_gap_duration_s
#     )
#     blink_array_pd_pp = filter_short_events(
#         blink_array_pd_pp, pp_params.short_event_min_len_s, label_mapping.blink
#     )
#     blink_events = blink_array_pd_pp.blink_events

#     onset_offset_events = get_onset_offset_events(event_array_pd)

#     return blink_events, onset_offset_events


def smooth_proba(proba: np.ndarray, pp_params: PPParams) -> np.ndarray:
    proba = proba.copy()
    proba_onset = proba[:, label_mapping.onset]
    proba_offset = proba[:, label_mapping.offset]

    proba_onset = smooth_array(proba_onset, pp_params.smooth_window)
    proba_offset = smooth_array(proba_offset, pp_params.smooth_window)
    proba_bg = 1 - np.sum([proba_onset, proba_offset], axis=0)

    proba[:, label_mapping.bg] = proba_bg
    proba[:, label_mapping.onset] = proba_onset
    proba[:, label_mapping.offset] = proba_offset
    return proba


def smooth_array(ary: np.ndarray, smooth_window: int = 1) -> np.ndarray:
    # Define mask and store as an array
    mask = np.ones((1, smooth_window)) / smooth_window
    mask = mask[0, :]
    # Convolve the mask with the raw data
    convolved_data = np.convolve(ary, mask, "same")
    return convolved_data


def classify(proba: np.ndarray, pp_params: PPParams) -> np.ndarray:
    pd_labels = np.argmax(proba, axis=1)
    pd_labels[
        proba[:, label_mapping.offset] > pp_params.proba_offset_threshold
    ] = label_mapping.offset
    pd_labels[
        proba[:, label_mapping.onset] > pp_params.proba_onset_threshold
    ] = label_mapping.onset
    return pd_labels


def get_onset_offset_events(array: EventArray) -> EventArray:
    onset_idx = np.where(array.labels == label_mapping.onset)[0]
    offset_idx = np.where(array.labels == label_mapping.offset)[0]

    onset_offset_array = EventArray(
        [array.start_times[0]], [array.end_times[-1]], [label_mapping.bg]
    )

    for start_time, end_time, label in zip(
        np.concatenate([array.start_times[onset_idx], array.start_times[offset_idx]]),
        np.concatenate([array.end_times[onset_idx], array.end_times[offset_idx]]),
        np.concatenate([array.labels[onset_idx], array.labels[offset_idx]]),
    ):
        onset_offset_array.insert_event(start_time, end_time, label)

    return onset_offset_array


def filter_wrong_sequence(array: EventArray, max_gap_duration_s=None) -> EventArray:
    none_idx = np.where(array.labels == label_mapping.bg)[0]
    onset_idx = np.where(array.labels == label_mapping.onset)[0]
    offset_idx = np.where(array.labels == label_mapping.offset)[0]

    keep_onset_idx = list(set(onset_idx) & set(offset_idx - 1))
    keep_offset_idx = [i + 1 for i in keep_onset_idx]

    # Discard blink events with long gap between onset and offset
    if max_gap_duration_s is not None:
        onset_idx_gap = list(set(onset_idx) & set(none_idx - 1) & set(offset_idx - 2))
        if onset_idx_gap:
            onset_idx_gap = np.asarray(onset_idx_gap)
            none = onset_idx_gap + 1
            onset_idx_gap = onset_idx_gap[array.duration_s[none] < max_gap_duration_s]
            keep_onset_idx += onset_idx_gap.tolist()
            keep_offset_idx += [i + 2 for i in onset_idx_gap]

    filtered_array = EventArray(
        [array.start_times[0]], [array.end_times[-1]], [label_mapping.bg]
    )
    for start_time, end_time in zip(
        array.start_times[keep_onset_idx], array.end_times[keep_offset_idx]
    ):
        filtered_array.insert_event(start_time, end_time, label_mapping.blink)
    return filtered_array


def filter_short_events(
    array: EventArray, min_len_s: float, select_label
) -> EventArray:
    """Remove short blinks from the sequence."""

    filtered_array = array.copy()
    mask_on = filtered_array.labels == select_label
    mask_short = filtered_array.duration_s < min_len_s
    del_mask = mask_on & mask_short
    filtered_array.remove_events(del_mask)
    return filtered_array
