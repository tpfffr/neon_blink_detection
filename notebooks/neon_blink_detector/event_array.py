from dataclasses import dataclass
import numpy as np
import typing as T
import copy


@dataclass
class BlinkEvent:
    start_time: int = None
    end_time: int = None
    label: str = None


class EventArray:
    def __init__(self, start_times, end_times, labels):
        assert len(start_times) == len(end_times) == len(labels)

        self.start_times = np.asarray(start_times)
        self.end_times = np.asarray(end_times)
        self.labels = np.asarray(labels)

    @property
    def blink_events(self) -> T.List[BlinkEvent]:
        blink_label = 3
        start_times = self.start_times[self.labels == blink_label]
        end_times = self.end_times[self.labels == blink_label]
        label_name = "Blink"
        return [
            BlinkEvent(start_time, end_time, label_name)
            for start_time, end_time in zip(start_times, end_times)
        ]

    @classmethod
    def from_samples(cls, timestamps, sample_labels, mapping=None):
        """Creates an instance from a time-series of labelled samples."""
        assert len(timestamps) == len(sample_labels)

        timestamps = np.asarray(timestamps)
        sample_labels = np.asarray(sample_labels)

        now_unequal_previous = sample_labels[1:] != sample_labels[:-1]
        now_unequal_previous = np.insert(now_unequal_previous, 0, True)

        start_times = timestamps[now_unequal_previous]
        labels = sample_labels[now_unequal_previous]

        end_times = np.roll(start_times, -1)
        end_times[-1] = timestamps[-1]

        if mapping:
            labels = np.array(list(map(lambda e: mapping[e], labels)))

        return cls(start_times, end_times, labels)

    def insert_event(self, start_time: float, end_time: float, label):

        overlapping = (self.start_times < end_time) & (self.end_times > start_time)
        overlapping_ind = np.nonzero(overlapping)[0]

        assert (
            len(overlapping_ind) == 1
        ), "Currently, can insert events only in the middle of existing events, i.e. when there is only 1 overlapping event."

        # cut the overlapping event in the middle and remember end time and label
        ind = overlapping_ind[0]
        previous_end_time = self.end_times[ind]
        previous_label = self.labels[ind]

        self.end_times[ind] = start_time  # cut

        # add two new events:
        #   - the requested event at the start and end time
        #   - a "fill event" which has has the same label as the event which has been
        #   cut and which fills the t0 to the next event
        self.start_times = np.insert(self.start_times, ind + 1, [start_time, end_time])
        self.end_times = np.insert(
            self.end_times, ind + 1, [end_time, previous_end_time]
        )
        self.labels = np.insert(self.labels, ind + 1, [label, previous_label])

    def combine_same_events(self) -> None:
        """Delete events which have the same label as the previous event."""

        equal_previous = self.labels[1:] == self.labels[:-1]
        equal_previous = np.insert(equal_previous, 0, False)
        self.labels = self.labels[~equal_previous]
        self.start_times = self.start_times[~equal_previous]

        # reconstruct end times from start times
        final_time = self.end_times[-1]
        self.end_times = np.roll(self.start_times, -1)
        self.end_times[-1] = final_time

    def remove_events(self, del_mask) -> None:
        # remember boundaries of time axis (start and end times of first and last event)
        initial_time = self.start_times[0]
        final_time = self.end_times[-1]

        # filter events
        self.labels = self.labels[~del_mask]
        self.start_times = self.start_times[~del_mask]

        self.combine_same_events()

        # fix events at the boundaries (if first or last event were removed)
        # last event must stop at the end of the time axis
        self.end_times[-1] = final_time
        # first event must start at the first samples
        self.start_times[0] = initial_time

    @property
    def duration_s(self):
        return self.end_times - self.start_times

    def copy(self):
        """Returns a copied instance of this object."""
        start_times = copy.deepcopy(self.start_times)
        end_times = copy.deepcopy(self.end_times)
        labels = copy.deepcopy(self.labels)
        return EventArray(start_times, end_times, labels)


def filter_wrong_sequence(array: EventArray, max_gap_duration_s=None) -> EventArray:
    """Filters out blink events with wrong sequence of labels and with  gaps between onset and offset longer than 'max_gap_duration_s'."""

    none_idx = np.where(array.labels == 0)[0]
    onset_idx = np.where(array.labels == 1)[0]
    offset_idx = np.where(array.labels == 2)[0]

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

    filtered_array = EventArray([array.start_times[0]], [array.end_times[-1]], [0])
    for start_time, end_time in zip(
        array.start_times[keep_onset_idx], array.end_times[keep_offset_idx]
    ):
        filtered_array.insert_event(start_time, end_time, 3)
    return filtered_array


def filter_short_events(
    array: EventArray, min_len_s: float, select_label
) -> EventArray:
    """Remove blinks shorter than 'min_len_s' from the sequence."""

    filtered_array = array.copy()
    mask_on = filtered_array.labels == select_label
    mask_short = filtered_array.duration_s < min_len_s
    del_mask = mask_on & mask_short
    filtered_array.remove_events(del_mask)
    return filtered_array
