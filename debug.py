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

        allowed_ids_set = set(str(x) for x in allowed_ids)
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


def debug_dataset_diagnostics(dataloader, cfg, batch_size=8):
    """
    Prints diagnostic information about the dataset, including filter configuration,
    valid conditions, and chunk counts vs theoretical maximums.

    Args:
        dataloader: The LongCycler or dict of dataloaders.
        cfg: The configuration object (OmegaConf or dict).
        batch_size: Batch size used for calculating expected batches (default: 8).
    """
    print("\n" + "=" * 40)
    print(" DEBUG: DATASET DIAGNOSTICS")
    print("=" * 40)

    # 1. Check the active Valid Condition
    # Access safely in case cfg structure varies slightly
    screen_config = cfg.dataset.modality_config.screen
    valid_condition = (
        screen_config.get("valid_condition", "Not Set")
        if hasattr(screen_config, "get")
        else getattr(screen_config, "valid_condition", "Not Set")
    )

    print(f"Active Valid Condition: {valid_condition}")

    # Check if filters are configured on the Screen modality
    has_filters = hasattr(cfg.dataset.modality_config.screen, "filters")
    print(f"Filter Configured: {has_filters}")

    # 2. Check individual session lengths
    loaders = dataloader.loaders if hasattr(dataloader, "loaders") else dataloader
    total_chunks = 0

    for key, loader in loaders.items():
        ds_len = len(loader.dataset)
        total_chunks += ds_len
        print(f"Session: {key:<50} | Valid Chunks: {ds_len}")

        dataset = loader.dataset
        valid_chunks = len(dataset)

        # Calculate Theoretical Max (Duration / Chunk Size)
        duration = dataset.end_time - dataset.start_time

        # Use screen sampling rate and chunk size from config
        screen_cfg = cfg.dataset.modality_config.screen
        stride_sec = screen_cfg.chunk_size / screen_cfg.sampling_rate

        max_chunks = int(duration / stride_sec)

        print(f"Session: {key:<20}")
        print(f"  > Valid Chunks (Filtered): {valid_chunks}")
        print(f"  > Theoretical Max (Raw):   {max_chunks}")

        if valid_chunks == max_chunks:
            print(
                "  !!! WARNING: Filter might be inactive (Counts match raw duration) !!!"
            )
        else:
            print(f"  > chunks removed by filter: {max_chunks - valid_chunks}")

    print("-" * 40)
    print(f"TOTAL CHUNKS: {total_chunks}")

    # Try to grab batch size from config if not provided explicitly, or use default
    if hasattr(cfg, "dataloader") and "batch_size" in cfg.dataloader:
        batch_size = cfg.dataloader.batch_size

    print(f"EXPECTED BATCHES (bs={batch_size}): {total_chunks // batch_size}")
    print("=" * 40 + "\n")
