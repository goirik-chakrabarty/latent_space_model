import json
import os

import numpy as np


def debug_filter_consistency(dataloader, allowed_session_ids_map):
    """
    Verifies that the dataset only contains time chunks belonging to the allowed IDs.

    Args:
        dataloader: The LongCycler or dict of dataloaders.
        allowed_session_ids_map: Dict {session_path: [list_of_allowed_ids]}
    """
    print("\n" + "=" * 60)
    print("DEBUG: Verifying Filtered Content (ID/Video Check)")
    print("=" * 60)

    issues_found = False

    # Handle both LongCycler (which has .loaders) and standard dicts
    loaders = dataloader.loaders if hasattr(dataloader, "loaders") else dataloader

    for data_key, loader in loaders.items():
        print(f"\nChecking Dataloader Session: '{data_key}'")

        # 1. Resolve the full path for this data_key to find the correct allowed list
        # We assume the data_key was derived from one of the paths in allowed_session_ids_map
        dataset_path = str(loader.dataset.root_folder)

        # Try to find exact path match
        allowed_ids = allowed_session_ids_map.get(dataset_path)

        # If not exact, try to match using the logic often used in experanto dataloaders
        if allowed_ids is None:
            for path, ids in allowed_session_ids_map.items():
                if str(path) == dataset_path:
                    allowed_ids = ids
                    break

        if allowed_ids is None:
            print(f"  [WARNING] Could not find allowed IDs for path: {dataset_path}")
            print(f"  Skipping verification for this session.")
            continue

        allowed_ids_set = set(allowed_ids)
        print(f"  Path: {dataset_path}")
        print(f"  Allowed IDs Count: {len(allowed_ids_set)}")

        # 2. Load Ground Truth Metadata
        meta_path = os.path.join(dataset_path, "screen", "combined_meta.json")
        timestamps_path = os.path.join(dataset_path, "screen", "timestamps.npy")

        if not os.path.exists(meta_path) or not os.path.exists(timestamps_path):
            print(f"  [ERROR] Metadata files missing. Cannot verify.")
            continue

        with open(meta_path, "r") as f:
            meta = json.load(f)
        timestamps = np.load(timestamps_path)

        # 3. Check the Dataset's Valid Times
        # _valid_screen_times are the start times of every valid chunk the dataset will yield
        valid_start_times = loader.dataset._valid_screen_times
        total_chunks = len(valid_start_times)

        if total_chunks == 0:
            print(f"  [WARNING] Dataset is empty! No valid chunks found.")
            continue

        print(f"  Total Valid Chunks generated: {total_chunks}")

        # 4. Monte Carlo Check: Sample random chunks to verify
        n_samples = min(10, total_chunks)  # Check 10 random chunks per session
        sample_indices = np.random.choice(total_chunks, size=n_samples, replace=False)

        passed_samples = 0

        for idx in sample_indices:
            t_start = valid_start_times[idx]

            # Find the frame index for this time
            # We use searchsorted to find insertion point
            frame_idx = np.searchsorted(timestamps, t_start)

            # Find which Image/Video ID owns this frame
            found_id = None
            for img_id, info in meta.items():
                # We assume the chunk starts strictly within the ID's duration
                start_f = info["first_frame_idx"]
                end_f = start_f + info["num_frames"]

                if start_f <= frame_idx < end_f:
                    found_id = img_id
                    break

            # Verification
            if found_id:
                if found_id in allowed_ids_set:
                    passed_samples += 1
                    # Optional: print(f"    [OK] Time {t_start:.2f}s -> ID {found_id}")
                else:
                    print(
                        f"    [FAIL] Time {t_start:.2f}s -> ID {found_id} (NOT in allowed list!)"
                    )
                    issues_found = True
            else:
                # If t_start maps to no ID, it might be a blank or inter-trial interval
                # Depending on strictness, you might flag this or ignore it.
                # Assuming here that chunks must belong to the selected IDs.
                print(
                    f"    [FAIL] Time {t_start:.2f}s -> No matching ID found in metadata."
                )
                issues_found = True

        print(
            f"  Result: {passed_samples}/{n_samples} sampled chunks passed verification."
        )

    print("\n" + "=" * 60)
    if issues_found:
        print("DEBUG RESULT: FAILURES FOUND. Dataset contains disallowed IDs.")
        # If failures found, verify keep_nans logic discussed previously
        print(
            "Hint: If non-allowed IDs are appearing, check if your 'nan_filter' or 'complement' logic is interfering."
        )
    else:
        print("DEBUG RESULT: SUCCESS. All sampled chunks belong to allowed IDs.")
    print("=" * 60 + "\n")


# --- Usage Example ---
# Place this right after you create your dataloader:
# debug_filter_consistency(train_dl, experiment_transfer)
