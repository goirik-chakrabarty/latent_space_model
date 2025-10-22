import numpy as np
import json

from experanto.datasets import (
    register_callable,  # need to import this to register the function
)
from experanto.interpolators import Interpolator
from experanto.intervals import (
    TimeInterval,
    find_complement_of_interval_array,
    uniquefy_interval_array,
)


@register_callable("filter1")
def nan_filter(vicinity=0.05):
    def implementation(device_: Interpolator, vicinity=vicinity):
        time_delta = device_.time_delta
        start_time = device_.start_time
        end_time = device_.end_time
        data = device_._data  # (T, n_neurons)

        # detect nans
        nan_mask = np.isnan(data)  # (T, n_neurons)
        nan_mask = np.any(nan_mask, axis=1)  # (T,)

        # Find indices where nan_mask is True
        nan_indices = np.where(nan_mask)[0]

        # Create invalid TimeIntervals around each nan point
        invalid_intervals = []
        vicinity_seconds = vicinity  # vicinity is already in seconds
        for idx in nan_indices:
            time_point = start_time + idx * time_delta
            interval_start = max(start_time, time_point - vicinity_seconds)
            interval_end = min(end_time, time_point + vicinity_seconds)
            invalid_intervals.append(TimeInterval(interval_start, interval_end))

        # Merge overlapping invalid intervals
        invalid_intervals = uniquefy_interval_array(invalid_intervals)

        # Find the complement of invalid intervals to get valid intervals
        valid_intervals = find_complement_of_interval_array(
            start_time, end_time, invalid_intervals
        )
        return valid_intervals

    return implementation


@register_callable("session_specific_id_filter")
def session_specific_id_filter(session_ids={}, complement=False):
    """
    Creates a filter that uses a different list of IDs for each session.

    Args:
        session_ids (dict): A dictionary mapping session paths to lists of IDs.
        complement (bool): If True, return the complement of the intervals.

    Returns:
        A function that can be used as a filter in the experanto library.
    """
    def implementation(device_: Interpolator,
                       session_path: str,
                       session_ids=session_ids,
                       complement=complement):


        id_list = session_ids.get(str(session_path), [])

        if not id_list:
            return []

        id_list = sorted(id_list)

        meta_path = f"{session_path}/screen/combined_meta.json"
        with open(meta_path, 'rb') as f:
            meta = json.load(f)

        if complement:
            all_ids = set(meta.keys())
            used_ids = set(id_list)
            complement_ids = sorted(all_ids - used_ids)
            # Create a temporary session_ids dict for the recursive call
            temp_session_ids = {session_path: complement_ids}
            return session_specific_id_filter(session_ids=temp_session_ids)(device_, session_path)

        timestamps = np.load(f"{session_path}/screen/timestamps.npy")

        intervals = []
        for i in range(len(id_list)):
            image_id = id_list[i]
            if image_id in meta:
                start = meta[image_id]['first_frame_idx']
                end = start + meta[image_id]['num_frames']
                intervals.append(TimeInterval(timestamps[start], timestamps[end]))

        return uniquefy_interval_array(intervals)

    return implementation