seed = 42
import sys

# sys.path.append("/srv/user/turishcheva/sensorium_replicate/sensorium_2023/")
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nnfabrik.utility.nn_helpers import set_random_seed
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal

set_random_seed(seed)
import os

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
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
from sensorium.utility.scores import get_correlations
from tqdm import tqdm

from experanto.configs import DEFAULT_CONFIG as cfg
from experanto.dataloaders import get_multisession_dataloader

pre_path_tr = "/mnt/vast-react/projects/neural_foundation_model/upsampling_without_hamming_30.0Hz/"
pre_path_test = "/mnt/vast-react/projects/neural_foundation_model/test_upsampling_without_hamming_30.0Hz/"

train = [
    "dynamic29156-11-10-Video-021a75e56847d574b9acbcc06c675055_30hz",
    "dynamic29228-2-10-Video-021a75e56847d574b9acbcc06c675055_30hz",
    "dynamic29234-6-9-Video-021a75e56847d574b9acbcc06c675055_30hz",
    "dynamic29513-3-5-Video-021a75e56847d574b9acbcc06c675055_30hz",
    "dynamic29514-2-9-Video-021a75e56847d574b9acbcc06c675055_30hz",
    "dynamic17797-8-5-Video-021a75e56847d574b9acbcc06c675055_30hz",
]

test_folder_scans = [
    "dynamic26872-17-20-Video-021a75e56847d574b9acbcc06c675055_30hz",
    "dynamic27204-5-13-Video-021a75e56847d574b9acbcc06c675055_30hz",
    # 'dynamic29515-10-12-Video-021a75e56847d574b9acbcc06c675055_30hz',
    # 'dynamic29623-4-9-Video-021a75e56847d574b9acbcc06c675055_30hz',
    # 'dynamic29647-19-8-Video-021a75e56847d574b9acbcc06c675055_30hz',
    # 'dynamic29712-5-9-Video-021a75e56847d574b9acbcc06c675055_30hz',
    # 'dynamic29755-2-8-Video-021a75e56847d574b9acbcc06c675055_30hz'
]

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
    """
    Compute the KL divergence D_KL(q || p) where:
    q(z) = N(mu, diag(sigma^2)) and p(z) = N(0, I)

    Parameters:
    - mu (torch.Tensor): Mean vector of the Gaussian distribution q(z), shape (B,time,hidden).
    - sigma (torch.Tensor): Standard deviation vector of q(z), not squared, shape (B,time,).

    Returns:
    - kl_div (torch.Tensor): The KL divergence.
    """

    sigma_squared = sigma**2
    inverse_sigma_squared = 1.0 / sigma_squared

    # Trace term: Sum of inverse variances (since trace of a diagonal matrix is the sum of its diagonal elements)
    dim = mu.shape[0] * mu.shape[1] * mu.shape[2]

    trace_term = (
        inverse_sigma_squared * dim
    )  # sigma is a scalar which is constant acrosss all dims

    # Quadratic term: (mu^T * Sigma_0^{-1} * mu)
    quadratic_term = torch.sum(mu**2) * inverse_sigma_squared

    # Log-determinant term: 2 * sum(log(sigma))
    log_det_term = (
        2 * torch.log(sigma) * dim
    )  # sigma is a scalar which is constant acrosss all dims

    kl_div = 0.5 * (trace_term + quadratic_term - dim + log_det_term)
    return kl_div


# compute exponentail moving average of correlation
def calculate_ema(data, alpha):
    ema = torch.zeros(len(data))
    ema[0] = data[0]

    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t - 1]

    return ema


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
    ema_span=0.3,  # ema for validation correlation
    scheduler_patience=6,  # patience for decaying learning rate
    latent=False,
    log_every_n_batch=50,
    **kwargs,
):
    """

    Args:
        model: model to be trained
        dataloaders: dataloaders containing the data to train the model with
        seed: random seed
        avg_loss: whether to average (or sum) the loss over a batch
        scale_loss: whether to scale the loss according to the size of the dataset
        loss_function: loss function to use
        stop_function: the function (metric) that is used to determine the end of the training in early stopping
        loss_accum_batch_n: number of batches to accumulate the loss over
        device: device to run the training on
        verbose: whether to print out a message for each optimizer step
        interval: interval at which objective is evaluated to consider early stopping
        patience: number of times the objective is allowed to not become better before the iterator terminates
        epoch: starting epoch
        lr_init: initial learning rate
        max_iter: maximum number of training iterations
        maximize: whether to maximize or minimize the objective function
        tolerance: tolerance for early stopping
        restore_best: whether to restore the model to the best state after early stopping
        lr_decay_steps: how many times to decay the learning rate after no improvement
        lr_decay_factor: factor to decay the learning rate with
        min_lr: minimum learning rate
        warmup_steps: number of batch steps for linear lr warump
        T_max: epoch periodicity of cosine annealing schedluer
        cb: whether to execute callback function
        zig : True if ZIG encoder is used as model
        k_reg: is a dictonary containg the fitted k_values for each mice, applies regularization to size of shape parameter k of gamma distribution if k_reg is not None but a dictionary,
        ema-Span: alpha factor of exponential moving avaerage of validation correlation
        **kwargs:

    Returns:

    """
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
        if not isinstance(model.core.regularizer(), tuple):
            regularizers = int(
                not detach_core
            ) * model.core.regularizer() + model.readout.regularizer(data_key)
        else:
            regularizers = int(not detach_core) * sum(
                model.core.regularizer()
            ) + model.readout.regularizer(data_key)
        if loss_function == "ZIGLoss":
            # one entry in a tuple corresponds to one paramter of ZIG
            # the output is (theta,k,loc,q)
            positions = None
            # args[0][0:1] removes behavior from the video input data.
            model_output = model(
                args[0].to(device),
                data_key=data_key,
                out_predicts=False,
                train=True,
                positions=positions,
                **kwargs,
            )
            theta = model_output[0]
            k = model_output[1]
            loc = model_output[2]
            q = model_output[3]
            time_left = k.shape[1]

            original_data = args[1].transpose(2, 1)[:, -time_left:, :].to(device)

            # create zero, non zero masks
            comparison_result = original_data >= loc
            nonzero_mask = comparison_result.int()

            comparison_result = original_data <= loc
            zero_mask = comparison_result.int()

            if k_regu:
                k_fitted = k_regu[data_key + "fitted_k"]
                # k is constant over time and has shape (batch_size,time,num_neurons)
                k_output = k[0, 0, :]
                # punish values of k that are far away from the fitted k value, scale the regularization to the size of zig loss
                k_regularized = ((k_fitted - k_output) ** 2).mean() * 7 * 10**7

            else:
                k_regularized = 0

            if (
                len(model_output) > 4
            ):  # that is the case only for the latent space model
                k = k.unsqueeze(-1)
                loc = loc.unsqueeze(-1)
                zero_mask = zero_mask.unsqueeze(-1)
                nonzero_mask = nonzero_mask.unsqueeze(-1)
                original_data = original_data.unsqueeze(-1)
                means = model_output[4]
                sigma_squared = model_output[5]
                n_samples = model_output[7]

                sigma = torch.sqrt(sigma_squared)
                # Mask neurons, which were given in Encoder
                neuron_mask = model_output[8].to(means.device)
                neuron_mask = neuron_mask.unsqueeze(-1).repeat(1, 1, 1, n_samples)
                zig_loss = (
                    -1
                    * loss_scale
                    * (
                        criterion(
                            theta,
                            k,
                            loc=loc,
                            q=q,
                            target=original_data,
                            zero_mask=zero_mask,
                            nonzero_mask=nonzero_mask,
                        )[0]
                    )
                )

                zig_loss.masked_fill_(~neuron_mask, 0)
                zig_loss = zig_loss.sum() + regularizers

                zig_loss = zig_loss / (
                    n_samples
                )  # zigloss is in that case an MC approximate for the mean of log p(y|x,z)
                # calculate KL divergence between Gaussian prior and approximate posterior

                kl_divergence = kl_divergence_gaussian(
                    means, sigma
                )  # * q.shape[2] #kl_divergence is constant across neuron dimension since latent is the same for all neurons

                differences = means[:, 1:] - means[:, :-1]
                neighbor_loss = torch.norm(differences, p=2, dim=2).sum()

                # average loss over batch_size and time
                zig_loss = (
                    1
                    / (means.shape[0] * means.shape[1])
                    * (zig_loss + 5 * kl_divergence)
                )  # loss is ElBO = p(y|x,z) + KL(q(z|x),p(z)), log_det=0 if no flow is applied

            else:
                zig_loss = (
                    -1
                    * loss_scale
                    * criterion(
                        theta,
                        k,
                        loc=loc,
                        q=q,
                        target=original_data,
                        zero_mask=zero_mask,
                        nonzero_mask=nonzero_mask,
                    )[0].sum()
                    + regularizers
                    + k_regularized
                )
            # only zig loss
            if len(model_output) > 4:
                return zig_loss, kl_divergence
            else:
                return zig_loss, None
            """
            return (
                -1*loss_scale
                * criterion(theta, k,
                            loc=loc, 
                            q=q, 
                            target=original_data, 
                            zero_mask=zero_mask, 
                            nonzero_mask=nonzero_mask)[0].sum()
                + regularizers
            )
            """
        else:
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

    ##### Model training ####################################################################################################
    model.to(device)
    set_random_seed(seed)
    model.train()
    if loss_function == "ZIGLoss":
        zig_loss_instance = zero_inflated_losses.ZIGLoss()
        criterion = zig_loss_instance.get_slab_logl
    else:
        criterion = getattr(modules, loss_function)(avg=avg_loss)
    stop_closure = partial(
        getattr(scores, stop_function),
        dataloaders=dataloaders["oracle"],
        device=device,
        per_neuron=False,
        avg=True,
        flow=model.flow,
        cell_coordinates=None,
    )

    n_iterations = len(dataloaders["train"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init)
    # Define the optimizer to only include parameters that require gradients
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_init)

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
    # set the number of iterations over which you would like to accummulate gradients
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
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=wandb_name,
            # Track hyperparameters and run metadata
            config={
                "learning_rate": lr_init,
                "architecture": wandb_model_config,
                "dataset": wandb_dataset_config,
                "cur_epochs": max_iter,
                "starting epoch": epoch,
                "lr_decay_steps": lr_decay_steps,
                "lr_decay_factor": lr_decay_factor,
                "min_lr": min_lr,
            },
        )

        wandb.define_metric(name="Epoch", hidden=True)
        wandb.define_metric(name="Batch", hidden=True)
        wandb.define_metric(name="Epoch - batched", hidden=True)
        wandb.define_metric(name="Batch - batched", hidden=True)

    print("wandb initialized")

    batch_no_tot = 0
    ema_values = []
    best_validation_correlation = 0
    # train over epochs
    for epoch, val_obj in early_stopping(
        model,
        stop_closure,
        interval=interval,
        patience=patience,
        start=epoch,
        max_iter=max_iter,
        maximize=maximize,
        tolerance=tolerance,
        restore_best=restore_best,
        scheduler=scheduler,
        lr_decay_steps=lr_decay_steps,
    ):

        # executes callback function if passed in keyword args
        if cb is not None:
            cb()

        # train over batches
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0
        epoch_val_loss = 0
        for batch_no, (data_key, batch) in tqdm(
            enumerate(dataloaders["train"]),
            total=n_iterations,
            desc="Epoch {}".format(epoch),
        ):
            batch_no_tot += 1
            # # TODO - polly, these two lines are basically the ones you want to change!
            # beh = torch.cat(
            #     [
            #         batch["eye_tracker"][:, :, :2].transpose(2, 1),
            #         batch["treadmill"][:, :, :2].transpose(2, 1),
            #     ],
            #     axis=1,
            # )
            # video = batch["screen"]
            # b_expanded = beh.unsqueeze(-1).unsqueeze(-1)  # or b[:, :, :, None, None]
            # # Now broadcast b to match the spatial dimensions [16, 3, 60, 144, 256]
            # beh_tiled = b_expanded.expand(-1, -1, -1, video.shape[3], video.shape[4])
            # # Concatenate along dim=1 to get [16, 4, 60, 144, 256]
            # video = torch.cat([video, beh_tiled], dim=1).to("cuda:0")

            # resp = batch["responses"].transpose(2, 1).to("cuda:0")

            # batch_kwargs = {
            #     "videos": video,
            #     # 'pupil_center_core': batch['eye_tracker'][:, :, 2:].transpose(2, 1).to('cuda:0'),
            #     "responses": resp,
            #     "pupil_center": batch["eye_tracker"][:, :, 2:]
            #     .transpose(2, 1)
            #     .to("cuda:0"),
            # }
            # batch_args = [video, resp, batch_kwargs["pupil_center"]]

            # Load video and responses, send to GPU
            video = batch["screen"].to("cuda:0")
            resp = batch["responses"].transpose(2, 1).to("cuda:0")

            # --- Create arguments for the model ---
            # batch_args are positional arguments for the model's forward()
            batch_args = [video, resp]

            # batch_kwargs are keyword arguments for the model's forward()
            batch_kwargs = {
                "videos": video,
                "responses": resp,
            }

            # # batch_args = list(data)
            # # batch_kwargs = data._asdict() if not isinstance(data, dict) else data
            # -----
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
            if (batch_no + 1) % log_every_n_batch == 0 and use_wandb:
                wandb_dict = {
                    "Epoch Train loss- batched": epoch_loss,
                    "Batch - batched": batch_no_tot,
                    "Epoch - batched": epoch,
                    "Learning rate - batched": optimizer.param_groups[0]["lr"],
                }
                wandb.log(wandb_dict)

        model.eval()
        ###
        lr = optimizer.param_groups[0]["lr"]
        ###
        ## after - epoch-analysis

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

        if save_checkpoints:
            if validation_correlation > best_validation_correlation:
                torch.save(model.state_dict(), f"{checkpoint_save_path}best.pth")
                best_validation_correlation = validation_correlation

        if loss_function == "PoissonLoss" or (not model.latent):
            val_loss, _ = full_objective(
                model,
                dataloaders["oracle"],
                data_key,
                *batch_args,
                **batch_kwargs,
                detach_core=detach_core,
            )
        else:
            val_loss, kl_div = full_objective(
                model,
                dataloaders["oracle"],
                data_key,
                *batch_args,
                **batch_kwargs,
                detach_core=detach_core,
            )

        # torch.save(
        # model.state_dict(), f"toymodels2/temp_save.pth"
        # )

        print(
            f"Epoch {epoch}, Batch {batch_no}, Train loss {loss}, Validation loss {val_loss}"
        )
        print(f"EPOCH={epoch}  validation_correlation={validation_correlation}")

        ema_values.append(validation_correlation)
        ema = calculate_ema(torch.tensor(ema_values), ema_span)[-1]
        if use_wandb:
            wandb_dict = {
                "Epoch Train loss": epoch_loss,
                "Batch": batch_no_tot,
                "Epoch": epoch,
                "validation_correlation": validation_correlation,
                # "log_det": log_det,
                "Epoch validation loss": val_loss,
                "EMA validation loss": ema,
                # "Poisson Loss": pos_loss,
                "ZIG Loss": val_loss,
                "Learning rate": lr,
            }
            wandb.log(wandb_dict)

        model.train()

    ##### Model evaluation ####################################################################################################
    model.eval()
    # if save_checkpoints:
    # torch.save(model.state_dict(), f"{checkpoint_save_path}final.pth")

    # Compute avg validation and test correlation
    validation_correlation = get_correlations(
        model,
        dataloaders["oracle"],
        device=device,
        as_dict=False,
        per_neuron=False,
        deeplake_ds=False,
        flow=model.flow,
        cell_coordinates=None,
    )
    print(f"\n\n FINAL validation_correlation {validation_correlation} \n\n")

    output = {}
    output["validation_corr"] = validation_correlation

    score = np.mean(validation_correlation)
    if use_wandb:
        wandb.finish()

    # removing the checkpoints except the last one
    to_clean = os.listdir(checkpoint_save_path)
    # to_clean = os.listdir("toymodels")
    for f2c in to_clean:
        if "epoch" in f2c:
            os.remove(os.path.join(checkpoint_save_path, f2c))

    return score


if __name__ == "__main__":

    full_paths = [f"{pre_path_tr}{t}/" for t in train] + [
        f"{pre_path_test}{t}/" for t in test_folder_scans
    ]

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

    for k in cfg.dataset.modality_config.keys():
        print(
            k,
            cfg.dataset.modality_config[k].sampling_rate,
            cfg.dataset.modality_config[k].chunk_size,
        )

    cfg["dataloader"]["prefetch_factor"] = 2
    cfg["dataloader"]["num_workers"] = 4
    cfg["dataloader"]["shuffle"] = True
    cfg["dataloader"]["pin_memory"] = False
    # cfg['dataset']['add_behavior_as_channels'] = True
    cfg["dataset"]["modality_config"]["screen"]["transforms"]["Resize"]["size"] = [
        36,
        64,
    ]
    cfg["dataset"]["modality_config"]["screen"]["sample_stride"] = cfg["dataset"][
        "modality_config"
    ]["screen"]["chunk_size"]
    print(f"stride: {cfg['dataset']['modality_config']['screen']['sample_stride']}")
    train_dl = get_multisession_dataloader(full_paths, cfg)

    mean_activity_dict = {}
    n_neurons_dict = {}
    data_keys = list(train_dl.loaders.keys())
    for k in data_keys:
        batch = next(iter(train_dl.loaders[k]))
        n_neurons_dict[k] = batch["responses"].shape[-1]
        mean_activity_dict[k] = (
            batch["responses"].reshape(-1, n_neurons_dict[k]).mean(axis=0)
        )

    # batch_size = batch['responses'].shape

    readout_dict = dict(
        bias=True,
        init_mu_range=0.2,
        init_sigma=1.0,
        gamma_readout=0.0,
        gauss_type="full",
        # grid_mean_predictor={
        #     "type": "cortex",
        #     "input_dimensions": 2,
        #     "hidden_layers": 1,
        #     "hidden_features": 30,
        #     "final_tanh": True,
        # },
        grid_mean_predictor=None,
        share_features=False,
        share_grid=False,
        shared_match_ids=None,
        gamma_grid_dispersion=0.0,
        zig=False,
        out_channels=1,
        kernel_size=(11, 5),
        batch_size=cfg["dataloader"]["batch_size"],
        # conv_out = conv_out
    )

    factorised_3d_model = make_video_model(
        None,
        seed,
        core_dict=factorised_3D_core_dict,
        core_type="3D_factorised",
        readout_dict=readout_dict.copy(),
        readout_type="gaussian",
        use_gru=False,
        gru_dict=None,
        use_shifter=False,  # set to True if behavior is included
        shifter_dict=None,
        shifter_type=None,
        deeplake_ds=False,
        n_neurons_dict=n_neurons_dict,
        mean_activity_dict=mean_activity_dict,
        experanto=True,
        readout_dim=factorised_3D_core_dict["hidden_channels"][-1],
    )

    latent = False

    cfg["dataset"]["modality_config"]["responses"]["sampling_rate"] = 30
    cfg["dataset"]["modality_config"]["responses"]["chunk_size"] = 60

    cfg["dataset"]["modality_config"]["eye_tracker"]["sampling_rate"] = 30
    cfg["dataset"]["modality_config"]["eye_tracker"]["chunk_size"] = 60

    cfg["dataset"]["modality_config"]["treadmill"]["sampling_rate"] = 30
    cfg["dataset"]["modality_config"]["treadmill"]["chunk_size"] = 60

    cfg["dataset"]["modality_config"]["screen"]["sampling_rate"] = 30
    cfg["dataset"]["modality_config"]["screen"]["chunk_size"] = 60

    cfg.dataset.modality_config.screen.valid_condition = {"tier": "validation"}
    cfg["dataset"]["modality_config"]["screen"]["sample_stride"] = cfg["dataset"][
        "modality_config"
    ]["screen"]["chunk_size"]

    dataloaders = {}
    dataloaders["train"] = train_dl
    # todo - undo it after the validation set labels are updated
    # dataloaders["oracle"] = val_dl
    dataloaders["oracle"] = {}
    for m in full_paths:
        dataloaders["oracle"][m.split("dynamic")[-1].split("-Video")[0]] = (
            get_multisession_dataloader([m], cfg)
        )

    lr_inint = 5e-3
    min_lr = 1e-5

    factorised_3d_model.to("cuda:0")

    validation_score = standard_trainer(
        factorised_3d_model,
        dataloaders,
        111,
        use_wandb=True,
        wandb_name="sensorium_model_8_mice_with_bs8_stride_80_4days",
        loss_function="PoissonLoss",
        # loss_function= "PoissonLoss",
        verbose=True,
        lr_decay_steps=4,
        lr_init=lr_inint,
        min_lr=min_lr,
        device=device,
        patience=12,  # 12#8,
        scheduler_patience=10,  # 10#6,
        checkpoint_save_path="./test_training/sensorium_model_8_mice_with_bs8_stride_80_4days/",
    )
