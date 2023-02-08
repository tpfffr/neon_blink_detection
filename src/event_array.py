import copy
import typing as T
from dataclasses import dataclass

import numpy as np

from src.label_mapper import label_mapping
from src.metrics import Scores, calculate_basic_scores
from utils import is_sorted


@dataclass
class BlinkEvent:
    start_time: int = None
    end_time: int = None
    label: str = None


class EventArray:
    def __init__(self, start_times, end_times, labels):
        assert len(start_times) == len(end_times) == len(labels)
        assert is_sorted(start_times)
        assert is_sorted(end_times)

        self.start_times = np.asarray(start_times)
        self.end_times = np.asarray(end_times)
        self.labels = np.asarray(labels)

    @property
    def blink_events(self) -> T.List[BlinkEvent]:
        label = label_mapping.blink
        start_times = self.start_times[self.labels == label]
        end_times = self.end_times[self.labels == label]
        label_name = label_mapping.legend[label]
        return [
            BlinkEvent(start_time, end_time, label_name)
            for start_time, end_time in zip(start_times, end_times)
        ]

    @property
    def duration_ns(self):
        return self.end_times - self.start_times

    @property
    def duration_s(self):
        return self.duration_ns / 1e9

    def copy(self):
        """Returns a copied instance of this object."""
        start_times = copy.deepcopy(self.start_times)
        end_times = copy.deepcopy(self.end_times)
        labels = copy.deepcopy(self.labels)
        return EventArray(start_times, end_times, labels)

    def insert_event(self, start_time: float, end_time: float, label):
        """Insert a new event into the array."""
        # find overlapping events
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

    @classmethod
    def from_samples(cls, timestamps, sample_labels, mapping=None):
        """Creates an instance from a time-series of labelled samples."""
        assert len(timestamps) == len(sample_labels)
        assert is_sorted(timestamps)

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

    def __len__(self):
        assert len(self.start_times) == len(self.end_times) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, item):
        return self.start_times[item], self.end_times[item], self.labels[item]

    def __iter__(self):
        return zip(self.start_times, self.end_times, self.labels)

    def to_list(self):
        """Print as list of dictionaries."""
        return [
            {
                "start_t": self.start_times[i],
                "end_t": self.end_times[i],
                "label": self.labels[i],
            }
            for i in range(len(self))
        ]

    def __eq__(self, other):
        return (
            isinstance(other, EventArray)
            and len(self) == len(other)
            and np.allclose(self.start_times, other.start_times)
            and np.allclose(self.end_times, other.end_times)
            and np.allclose(self.labels, other.labels)
        )


class MatchedEventArray:
    def __init__(
        self,
        array_gt: EventArray,
        array_pd: EventArray,
        label_on,
        label_bg,
        iou_thr: float,
    ):
        self.iou_thr = iou_thr
        self.label_on = label_on
        self.label_bg = label_bg
        self.array_gt = self._fix_array_blink_label(array_gt)
        self.array_pd = self._fix_array_blink_label(array_pd)

        self.iou_matrix = self._get_iou_matrix()
        self._set_matches()

    def _fix_array_blink_label(self, old_array: EventArray) -> EventArray:
        array = old_array.copy()
        array.labels[array.labels != self.label_on] = self.label_bg
        array.combine_same_events()
        return array

    def _get_iou_matrix(self) -> np.ndarray:
        """Constructs a matrix which contains the IoU scores of each event in the
        ground truth array with each event in the predicted array.
        The matrix shape is (N_events_gt, N_events_pd)"""

        overlap = self._get_overlap_matrix()
        union = self._get_union_matrix()
        return overlap / union

    def _get_overlap_matrix(self) -> np.ndarray:
        """Constructs a matrix containing the overlap of each event in the ground truth
        array with each event in the predicted array in seconds.
        The matrix shape is (N_events_gt, N_events_pd)"""

        start_of_overlap = np.maximum(
            self.array_gt.start_times[:, np.newaxis],
            self.array_pd.start_times[np.newaxis, :],
        )
        end_of_overlap = np.minimum(
            self.array_gt.end_times[:, np.newaxis],
            self.array_pd.end_times[np.newaxis, :],
        )
        overlap = end_of_overlap - start_of_overlap
        overlap[overlap <= 0] = 0
        return overlap

    def _get_union_matrix(self) -> np.ndarray:
        """Constructs a matrix containing the maximum minus minimum time of each event
        in the ground truth array with each event in the predicted array in seconds.
        The matrix shape is (N_events_gt, N_events_pd)"""

        max_end_times = np.maximum(
            self.array_gt.end_times[:, np.newaxis],
            self.array_pd.end_times[np.newaxis, :],
        )
        min_start_times = np.minimum(
            self.array_gt.start_times[:, np.newaxis],
            self.array_pd.start_times[np.newaxis, :],
        )
        union = max_end_times - min_start_times
        return union

    def _set_matches(self) -> None:
        """Performs the event matching.
        Match each gt array with the first pred array which has enough IoU"""

        # threshold IoU
        over_thr = self.iou_matrix * (self.iou_matrix > self.iou_thr)
        over_thr[over_thr == 0] = -1

        # indicates whether any event could be matched
        found_match = np.any(over_thr > 0, axis=1)
        # if there is a match, indicates the index of the match
        ind_match = np.nanargmax(over_thr, axis=1)
        # indicates match index, or (-1) if not match is found
        ind_match[~found_match] = -1

        # match only the first event, if multiple events can be matched
        equal_previous = ind_match[1:] == ind_match[:-1]
        equal_previous = np.insert(equal_previous, 0, False)
        # match only the first event, if multiple events can be matched
        ind_match[equal_previous] = -1

        # there might be some duplicates remaining, because sometimes several
        # non-subsequent events are matched to the same. Delete those too, only
        # accept the first occurrence of a match.
        self._remove_duplicates(ind_match)

        # get inverse mapping (which maps indices from predicted to ground truth array)
        ind_match_inverse = np.ones(len(self.array_pd), dtype=np.int) * (-1)
        for i_gt, i_pd in enumerate(ind_match):
            if i_pd >= 0:
                ind_match_inverse[i_pd] = i_gt

        # indices of matched predicted events for each gt event
        self.ind_match = ind_match
        # indices of matched gt events for each predicted event
        self.ind_match_inverse = ind_match_inverse

    @staticmethod
    def _remove_duplicates(array) -> None:
        """Finds all duplicates in an array and replace all but the first ocurrence
        with -1.
        """
        unique, index, count = np.unique(
            array, axis=0, return_index=True, return_counts=True
        )
        for i, u in enumerate(unique):

            if u == -1:
                continue

            if count[i] > 1:
                find_others = np.nonzero(array == u)[0]
                find_others = find_others[1:]
                array[find_others] = -1

    def _build_eval_pairs(self) -> np.ndarray:
        """Builds an (N,2) array which maps ground truth labels onto predicted labels.
        -1 will be used if the events where not mapped.
        This array serves as the basis for the confusion matrix used for scoring.
        """

        # build evaluation array for all gt events
        # (containing all correct and incorrect pairs)
        match_labels = np.full(self.array_gt.labels.shape, -1)
        found_match = np.where(self.ind_match >= 0)[0]
        match_labels[found_match] = self.array_pd.labels[self.ind_match[found_match]]

        matched_pairs = np.stack((self.array_gt.labels, match_labels), axis=1)

        # add unmatched predicted events to the evaluation array
        unmatched_pd = np.where(self.ind_match_inverse == -1)[0]
        unmatched_pd_labels = self.array_pd.labels[unmatched_pd]

        unmatched_pairs = np.stack(
            (np.full(unmatched_pd_labels.shape, -1), unmatched_pd_labels), axis=1
        )
        eval_pairs = np.vstack((matched_pairs, unmatched_pairs))
        return eval_pairs

    def get_RTO_RTD_scores(self) -> dict:
        """Get relative timing offset (RTO) and deviation (RTD)
        for onset and offset of events."""

        pairs = [
            (i_gt, i_pd)
            for i_gt, i_pd in enumerate(self.ind_match)
            if (self.array_gt.labels[i_gt] == self.label_on)
            and (self.array_pd.labels[i_pd] == self.label_on)
            and self.ind_match[i_gt] != -1
        ]

        # get on-/offset times
        if pairs:
            pair_gt, pair_pd = np.array(pairs).T
            onset_times_gt = self.array_gt.start_times[pair_gt]
            onset_times_pd = self.array_pd.start_times[pair_pd]
            offset_times_gt = self.array_gt.end_times[pair_gt]
            offset_times_pd = self.array_pd.end_times[pair_pd]
            duration_gt = self.array_gt.duration_s[pair_gt]
            duration_pd = self.array_pd.duration_s[pair_pd]

            # compute RTO/RTD
            RTO_onset = np.abs(onset_times_gt - onset_times_pd).mean()
            RTD_onset = np.abs(onset_times_gt - onset_times_pd).std()
            RTO_offset = np.abs(offset_times_gt - offset_times_pd).mean()
            RTD_offset = np.abs(offset_times_gt - offset_times_pd).std()
        else:
            RTO_onset = np.nan
            RTD_onset = np.nan
            RTO_offset = np.nan
            RTD_offset = np.nan
            duration_gt = []
            duration_pd = []

        scores = {
            "RTO_onset": RTO_onset,
            "RTD_onset": RTD_onset,
            "RTO_offset": RTO_offset,
            "RTD_offset": RTD_offset,
            "duration_gt": duration_gt,
            "duration_pd": duration_pd,
        }
        return scores

    def get_average_IOU_score(self) -> dict:
        """Get average IOU score for a matched sequence.
        Counts only ground-truth blinks and counts non-matched ones as having IoU=0."""

        pairs = [
            (i_gt, i_pd)
            for i_gt, i_pd in enumerate(self.ind_match)
            if (self.array_gt.labels[i_gt] == self.label_on)
        ]
        pairs = np.array(pairs)
        iou = [self.iou_matrix[i_gt, i_pd] for (i_gt, i_pd) in pairs if i_pd >= 0]
        return {"mean_IoU": np.mean(iou), "IoU": iou}

    def get_scores(self) -> Scores:
        eval_pairs = self._build_eval_pairs()
        y_true, y_pred = eval_pairs.T
        scores = calculate_basic_scores(y_true, y_pred, self.label_on, self.label_bg)
        scores.replace(**self.get_RTO_RTD_scores())
        scores.replace(**self.get_average_IOU_score())
        return scores


class Samples:
    def __init__(self, timestamps: T.Sequence, labels: T.Sequence):
        assert len(timestamps) == len(labels)
        # assert is_sorted(timestamps)

        self.timestamps = np.array(timestamps, copy=True)
        self.labels = np.array(labels, copy=True)

    @property
    def event_array(self) -> EventArray:
        return EventArray.from_samples(self.timestamps, self.labels)

    @property
    def blink_array(self) -> EventArray:
        return EventArray.from_samples(self.timestamps, self.blink_labels)

    @property
    def blink_labels(self) -> np.ndarray:
        labels = np.array(self.labels, copy=True)
        labels[labels == label_mapping.onset] = label_mapping.blink
        labels[labels == label_mapping.offset] = label_mapping.blink
        return labels

    def __len__(self):
        assert len(self.timestamps) == len(self.labels)
        return len(self.timestamps)

    @property
    def n_onset(self) -> int:
        return len(self.labels[self.labels == label_mapping.onset])

    @property
    def n_offset(self) -> int:
        return len(self.labels[self.labels == label_mapping.offset])
