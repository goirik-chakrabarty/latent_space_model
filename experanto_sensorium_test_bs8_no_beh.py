import argparse
import json
import os
import random
import sys

# sys.path.append("/srv/user/turishcheva/sensorium_replicate/sensorium_2023/")
from functools import partial
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from eval import eval_model
from moments import load_mean_variance
from neuralpredictors.layers.cores.conv2d import Stacked2dCore
from neuralpredictors.layers.encoders.mean_variance_functions import fitted_zig_mean
from neuralpredictors.layers.encoders.zero_inflation_encoders import ZIGEncoder
from neuralpredictors.measures import modules, zero_inflated_losses
from neuralpredictors.training import early_stopping
from nnfabrik.utility.nn_helpers import set_random_seed
from sensorium.datasets.mouse_video_loaders import mouse_video_loader
from sensorium.models.make_model import make_video_model
from sensorium.utility import scores
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
import common_filters
from sensorium.utility.scores import get_correlations
from tqdm import tqdm

from experanto.configs import DEFAULT_CONFIG as cfg
from experanto.dataloaders import (
    get_multisession_concat_dataloader,
    get_multisession_dataloader,
)

# --- CONFIGURATION (Preserved from training script) ---
factorised_3D_core_dict = dict(
    input_channels=1,
    hidden_channels=[32, 64, 128],
    spatial_input_kernel=(11, 11),
    temporal_input_kernel=11,
    spatial_hidden_kernel=(5, 5),
    temporal_hidden_kernel=5,
    stride=1,
    layers=3,
    gamma_input_spatial=10,
    gamma_input_temporal=0.01,
    bias=True,
    hidden_nonlinearities="elu",
    x_shift=0,
    y_shift=0,
    batch_norm=True,
    laplace_padding=None,
    input_regularizer="LaplaceL2norm",
    padding=False,
    final_nonlin=True,
    momentum=0.7,
)
shifter_dict = dict(
    gamma_shifter=0,
    shift_layers=3,
    input_channels_shifter=2,
    hidden_channels_shifter=5,
)


def kl_divergence_gaussian(mu, sigma):
    sigma_squared = sigma**2
    inverse_sigma_squared = 1.0 / sigma_squared
    dim = mu.shape[0] * mu.shape[1] * mu.shape[2]
    trace_term = inverse_sigma_squared * dim
    quadratic_term = torch.sum(mu**2) * inverse_sigma_squared
    log_det_term = 2 * torch.log(sigma) * dim
    kl_div = 0.5 * (trace_term + quadratic_term - dim + log_det_term)
    return kl_div


def calculate_ema(data, alpha):
    ema = torch.zeros(len(data))
    ema[0] = data[0]
    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t - 1]
    return ema


# --- STANDARD TRAINER (Preserved from training script) ---
def standard_trainer(
    model,
    dataloaders,
    seed,
    avg_loss=False,
    scale_loss=True,
    loss_function="PoissonLoss",
    stop_function="get_correlations",
    loss_accum_batch_n=None,
    device="cuda",
    verbose=True,
    interval=1,
    patience=5,
    epoch=0,
    lr_init=0.005,
    max_iter=200,
    maximize=True,
    tolerance=1e-6,
    restore_best=True,
    lr_decay_steps=3,
    lr_decay_factor=0.3,
    min_lr=0.0001,
    cb=None,
    detach_core=False,
    use_wandb=True,
    wandb_project="sensorioum-baseline",
    wandb_entity="sinzlab",
    wandb_name=None,
    wandb_model_config=None,
    wandb_dataset_config=None,
    save_checkpoints=True,
    checkpoint_save_path="local/",
    chpt_save_step=15,
    k_reg=False,
    ema_span=0.3,
    scheduler_patience=6,
    latent=False,
    log_every_n_batch=50,
    **kwargs,
):
    print(loss_function)

    def full_objective(model, dataloader, data_key, *args, k_regu=k_reg, **kwargs):
        if isinstance(dataloader, dict):
            loss_scale = (
                np.sqrt(
                    len(dataloader[data_key].loaders[data_key].dataset)
                    / args[0].shape[0]
                )
                if scale_loss
                else 1.0
            )
        else:
            loss_scale = (
                np.sqrt(len(dataloader.loaders[data_key].dataset) / args[0].shape[0])
                if scale_loss
                else 1.0
            )

        # --- MODIFICATION FOR TEST: FREEZE CORE ---
        # If detach_core is True, we ensure regularizers only come from readout
        if detach_core:
            regularizers = model.readout.regularizer(data_key)
        else:
            if not isinstance(model.core.regularizer(), tuple):
                regularizers = int(
                    not detach_core
                ) * model.core.regularizer() + model.readout.regularizer(data_key)
            else:
                regularizers = int(not detach_core) * sum(
                    model.core.regularizer()
                ) + model.readout.regularizer(data_key)

        model_output = model(args[0].to(device), data_key=data_key, **kwargs)
        time_left = model_output.shape[1]
        original_data = args[1].transpose(2, 1)[:, -time_left:, :].to(device)

        return (
            loss_scale
            * criterion(
                model_output,
                original_data,
            )
            + regularizers
        ), None

    model.to(device)
    model.train()

    criterion = getattr(modules, loss_function)(avg=avg_loss)

    n_iterations = len(dataloaders["train"])

    # --- MODIFICATION: OPTIMIZER FOR TRANSFER ---
    # Only optimize parameters that require gradients (Readout)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr_init
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if maximize else "min",
        factor=lr_decay_factor,
        patience=scheduler_patience,
        threshold=tolerance,
        min_lr=min_lr,
        verbose=verbose,
        threshold_mode="abs",
    )

    optim_step_count = (
        len(dataloaders["train"].loaders.keys())
        if loss_accum_batch_n is None
        else loss_accum_batch_n
    )
    print(f"optim_step_count = {optim_step_count}")

    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            config={"learning_rate": lr_init, "type": "transfer_learning"},
        )

    batch_no_tot = 0
    ema_values = []
    best_validation_correlation = 0
    calculate_val_loss = True

    for epoch in range(max_iter):
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0

        for batch_no, (data_key, batch) in tqdm(
            enumerate(dataloaders["train"]),
            total=n_iterations,
            desc="Epoch {}".format(epoch),
        ):
            batch_no_tot += 1

            # --- NO BEHAVIOR SETUP ---
            video = batch["screen"].to("cuda:0")
            resp = batch["responses"].transpose(2, 1).to("cuda:0")

            batch_args = [video, resp]
            batch_kwargs = {
                "videos": video,
                "responses": resp,
            }

            loss = full_objective(
                model,
                dataloaders["train"],
                data_key,
                *batch_args,
                **batch_kwargs,
                detach_core=detach_core,
            )[0]

            loss.backward()
            epoch_loss += loss.detach()

            if (batch_no + 1) % optim_step_count == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        lr = optimizer.param_groups[0]["lr"]

        if calculate_val_loss:
            validation_correlation = get_correlations(
                model,
                dataloaders["oracle"],
                device=device,
                as_dict=False,
                per_neuron=False,
                deeplake_ds=False,
                flow=model.flow,
                cell_coordinates=None,
                behavioral_modulation=False,
            )

            # --- Early Stopping / Checkpointing ---
            scheduler.step(validation_correlation)

            if validation_correlation > best_validation_correlation:
                best_validation_correlation = validation_correlation
                # Optional: Save best readout
                # torch.save(model.state_dict(), f"{checkpoint_save_path}best_transfer.pth")

            val_loss, _ = full_objective(
                model,
                dataloaders["oracle"],
                data_key,
                *batch_args,
                **batch_kwargs,
                detach_core=detach_core,
            )

        print(f"Epoch {epoch}, Batch {batch_no}, Train loss {loss}")
        if calculate_val_loss:
            print(
                f"EPOCH={epoch}  validation_correlation={validation_correlation} validation_loss={val_loss}"
            )
            ema_values.append(validation_correlation)
            ema = calculate_ema(torch.tensor(ema_values), ema_span)[-1]

        if use_wandb:
            wandb.log(
                {
                    "Epoch Train loss": epoch_loss,
                    "validation_correlation": validation_correlation,
                    "Epoch validation loss": val_loss,
                    "EMA validation loss": ema,
                    "Learning rate": lr,
                }
            )

        model.train()

    model.eval()
    if calculate_val_loss:
        validation_correlation = get_correlations(
            model,
            dataloaders["oracle"],
            device=device,
            as_dict=False,
            per_neuron=False,
            deeplake_ds=False,
            flow=model.flow,
            cell_coordinates=None,
            behavioral_modulation=False,
        )
        print(f"\n\n FINAL validation_correlation {validation_correlation} \n\n")
        score = np.mean(validation_correlation)

    if use_wandb:
        wandb.finish()

    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pre-trained model checkpoint",
    )
    parser.add_argument("--experiment_name", type=str, default="test_transfer")
    args = parser.parse_args()

    set_random_seed(42)
    seed = 42

    # --- DATA SPLITTING LOGIC (Preserved) ---
    save_path = "trial_details/optimal_trial_details_with_split.json"
    raw_path = "trial_details/optimal_trial_details.json"

    # We strictly use the "test" split defined in the original script for transfer
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            trial_details = json.load(f)
        session_specific_ids_test = trial_details.get("session_specific_ids_test", {})
    else:
        # Fallback if split file doesn't exist (copying logic from original)
        with open(raw_path, "r") as f:
            data = json.load(f)
        df_A = pd.DataFrame(data["dataset_A_common_trials"])
        df_B = pd.DataFrame(data["dataset_B_unique_trials"])
        set_A = set(df_A.session)
        set_B = set(df_B.session)
        common_sessions = list(set_A.intersection(set_B))
        test_folder_scans = random.sample(
            common_sessions, k=(len(common_sessions) // 5)
        )

        pre_path_tr = "/mnt/vast-react/projects/neural_foundation_model/upsampling_without_hamming_30.0Hz"

        session_specific_ids_test_A = {
            os.path.join(pre_path_tr, session): df_A[
                df_A.session == session
            ].trial_idx.tolist()
            for session in test_folder_scans
        }
        session_specific_ids_test_B = {
            os.path.join(pre_path_tr, session): df_B[
                df_B.session == session
            ].trial_idx.tolist()
            for session in test_folder_scans
        }
        session_specific_ids_test = {}
        for session in test_folder_scans:
            ids_A = session_specific_ids_test_A.get(
                os.path.join(pre_path_tr, session), []
            )
            ids_B = session_specific_ids_test_B.get(
                os.path.join(pre_path_tr, session), []
            )
            combined_ids = list(set(ids_A + ids_B))
            session_specific_ids_test[os.path.join(pre_path_tr, session)] = combined_ids

    # --- DATALOADER PREPARATION ---
    # The 'experiment' here is the TEST set from original, used for TRANSFER training
    experiment_transfer = session_specific_ids_test
    print(f"Transfer Learning on {len(experiment_transfer)} sessions.")

    # Truncation logic (from original)
    experiment_transfer = dict(list(experiment_transfer.items())[:])
    for k in experiment_transfer.keys():
        # Using same truncation logic for consistency, or remove if you want full test set
        n = len(experiment_transfer[k])
        experiment_transfer[k] = experiment_transfer[k][4 * n // 6 :]
        print(f"Session {k}: {len(experiment_transfer[k])} trials")

    # Config updates (from original)
    cfg["dataset"]["modality_config"]["responses"]["sampling_rate"] = 30
    cfg["dataset"]["modality_config"]["responses"]["chunk_size"] = 80
    cfg["dataset"]["modality_config"]["eye_tracker"]["sampling_rate"] = 30
    cfg["dataset"]["modality_config"]["eye_tracker"]["chunk_size"] = 80
    cfg["dataset"]["modality_config"]["treadmill"]["sampling_rate"] = 30
    cfg["dataset"]["modality_config"]["treadmill"]["chunk_size"] = 80
    cfg["dataset"]["modality_config"]["screen"]["sampling_rate"] = 30
    cfg["dataset"]["modality_config"]["screen"]["chunk_size"] = 80
    cfg["dataset"]["modality_config"]["screen"]["transforms"]["normalization"] = {
        "mean": 113,
        "std": 59,
    }
    cfg["dataloader"]["batch_size"] = 8
    cfg["dataloader"]["prefetch_factor"] = 2
    cfg["dataloader"]["num_workers"] = 1
    cfg["dataloader"]["shuffle"] = True
    cfg["dataloader"]["pin_memory"] = False
    cfg["dataset"]["modality_config"]["screen"]["transforms"]["Resize"]["size"] = [
        36,
        64,
    ]
    cfg["dataset"]["modality_config"]["screen"]["sample_stride"] = cfg["dataset"][
        "modality_config"
    ]["screen"]["chunk_size"]

    # Filter logic
    cfg.dataset.modality_config.treadmill.filters.custom_interval_filter = {
        "__key__": "session_specific_id_filter",
        "session_ids": experiment_transfer,
    }

    # Create Transfer Dataloader
    print("Creating Transfer Dataloaders...")
    train_dl = get_multisession_dataloader(list(experiment_transfer.keys()), cfg)

    # Calculate statistics for the NEW readout
    mean_activity_dict = {}
    n_neurons_dict = {}
    data_keys = list(train_dl.loaders.keys())
    for k in data_keys:
        batch = next(iter(train_dl.loaders[k]))
        n_neurons_dict[k] = batch["responses"].shape[-1]
        mean_activity_dict[k] = (
            batch["responses"].reshape(-1, n_neurons_dict[k]).mean(axis=0)
        )

    readout_dict = dict(
        bias=True,
        init_mu_range=0.2,
        init_sigma=1.0,
        gamma_readout=0.0,
        gauss_type="full",
        grid_mean_predictor=None,
        share_features=False,
        share_grid=False,
        shared_match_ids=None,
        gamma_grid_dispersion=0.0,
        zig=False,
        out_channels=1,
        kernel_size=(11, 5),
        batch_size=cfg["dataloader"]["batch_size"],
    )

    # --- MODEL BUILD ---
    print("Building Model...")
    factorised_3d_model = make_video_model(
        None,
        seed,
        core_dict=factorised_3D_core_dict,
        core_type="3D_factorised",
        readout_dict=readout_dict.copy(),
        readout_type="gaussian",
        use_gru=False,
        gru_dict=None,
        use_shifter=False,  # No behavior
        shifter_dict=None,
        shifter_type=None,
        deeplake_ds=False,
        n_neurons_dict=n_neurons_dict,  # Using dimensions from TEST set
        mean_activity_dict=mean_activity_dict,
        experanto=True,
        readout_dim=factorised_3D_core_dict["hidden_channels"][-1],
    )

    # --- LOAD PRE-TRAINED WEIGHTS & FREEZE ---
    print(f"Loading weights from {args.model_path}")
    pretrained_state = torch.load(args.model_path, map_location=device)

    # strict=False allows loading Core weights while ignoring the mismatch in Readout layers
    missing, unexpected = factorised_3d_model.load_state_dict(
        pretrained_state, strict=False
    )
    print(f"Weights loaded. Missing keys (Readout): {len(missing)}")

    print("Freezing Core parameters...")
    for name, param in factorised_3d_model.named_parameters():
        if "core" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True  # Readout is trainable

    factorised_3d_model.to(device)

    # --- PREPARE DATALOADERS DICT ---
    # For transfer, we train on the 'test' set.
    # Ideally, we should split this into train/val, but following the simple test pattern,
    # we use the same loader for oracle or a subset if available.
    dataloaders = {}
    dataloaders["train"] = train_dl

    # Creating a separate oracle loader (same data, just for metric calculation)
    dataloaders["oracle"] = {}
    for m in experiment_transfer.keys():
        dataloaders["oracle"][m.split("dynamic")[-1].split("-Video")[0]] = (
            get_multisession_dataloader([m], cfg)
        )

    # --- RUN TRANSFER TRAINING ---
    lr_inint = 5e-3
    min_lr = 1e-5

    print("Starting Transfer Learning...")
    validation_score = standard_trainer(
        factorised_3d_model,
        dataloaders,
        seed=seed,
        use_wandb=True,
        wandb_name=f"transfer_{args.experiment_name}",
        loss_function="PoissonLoss",
        verbose=True,
        lr_decay_steps=4,
        lr_init=lr_inint,
        min_lr=min_lr,
        device=device,
        patience=12,
        scheduler_patience=10,
        checkpoint_save_path="./test_transfer_checkpoints/",
        detach_core=True,  # Explicit flag for the trainer
    )

    print(f"Transfer Learning Complete. Score: {validation_score}")
