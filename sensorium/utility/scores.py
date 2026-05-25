import random
import warnings

import numpy as np
import torch
import torch.distributions as dist
from neuralpredictors.measures.np_functions import corr
from neuralpredictors.training import device_state


def model_predictions(
    model,
    dataloader,
    data_key,
    device="cuda:0",
    skip=50,
    deeplake_ds=False,
    prior=False,
    recursive=False,
    repeats=10,
    n_samples=100,
    flow=False,
    dropout_prob=None,
    cell_coordinates=None,
    behavioral_modulation=False,
):
    """
    computes model predictions for a given dataloader and a model
    Returns:
        target: ground truth, i.e. neuronal firing rates of the neurons
        output: responses as predicted by the network
    """

    target, output = [], []
    neuron_mask = None

    for _, batch in dataloader:

        video = batch["screen"]
        responses = batch["responses"].transpose(2, 1)
        with torch.no_grad():
            resp = responses.detach().cpu()[:, :, skip:]
            target = target + list(resp)
            with device_state(model, device):
                responses = responses.to(device)
                if behavioral_modulation:
                    beh = torch.cat(
                        [
                            batch["eye_tracker"][:, :, :2].transpose(2, 1),
                            batch["treadmill"][:, :, :2].transpose(2, 1),
                        ],
                        axis=1,
                    )
                    b_expanded = beh.unsqueeze(-1).unsqueeze(-1)
                    beh_tiled = b_expanded.expand(-1, -1, -1, video.shape[3], video.shape[4])
                    video = torch.cat([video, beh_tiled], dim=1).to(device)
                    batch_kwargs = {
                        "videos": video,
                        "pupil_center_core": batch["eye_tracker"][:, :, 2:]
                        .transpose(2, 1)
                        .to(device),
                        "responses": responses,
                        "pupil_center": batch["eye_tracker"][:, :, 2:]
                        .transpose(2, 1)
                        .to(device),
                    }
                else:
                    video = video.to(device)
                    batch_kwargs = {
                        "videos": video,
                        "responses": responses,
                    }
                images = video
                out_predicts = True
                if model.position_features:
                    positions = cell_coordinates[data_key]
                else:
                    positions = None

                if prior:
                    out = model.forward_prior(
                        images,
                        data_key=data_key,
                        out_predicts=out_predicts,
                        repeats=repeats,
                        n_samples=n_samples,
                        positions=positions,
                        **batch_kwargs,
                    )
                else:
                    if dropout_prob and (
                        not neuron_mask
                    ):  # neuron_mask should be same for all batches
                        n_neurons = responses.shape[1]
                        n_neurons_masked = (
                            int(n_neurons * (dropout_prob - 0.25)) + 1
                        )  # first quarter is always masked
                        neuron_mask = random.sample(
                            range(n_neurons // 4, n_neurons), n_neurons_masked
                        )
                    # bug is somewhere here?
                    out = model(
                        images,
                        data_key=data_key,
                        neuron_mask=neuron_mask,
                        out_predicts=out_predicts,
                        positions=positions,
                        **batch_kwargs,
                    )
                out = out.detach().cpu()[:, -resp.shape[-1] :, :]

                assert (
                    out.shape[1] == resp.shape[-1]
                ), f"model prediction is too short ({out.shape[1]} vs {resp.shape[-1]})"
                output = output + list(out.permute(0, 2, 1))
    return target, output


def get_correlations(
    model,
    dataloaders,
    tier=None,
    device="cpu",
    as_dict=False,
    per_neuron=True,
    deeplake_ds=False,
    masking=True,
    forward_prior=False,
    forward_recursive=False,
    repeats=10,
    n_samples=100,
    flow=False,
    dropout_prob=None,
    cell_coordinates=None,
    behavioral_modulation=False,
    **kwargs,
):
    """
    Computes single-trial correlation between model prediction and true responses
    Args:
        model (torch.nn.Module): Model used to predict responses.
        dataloaders (dict): dict of test set torch dataloaders.
        tier(str): the data-tier (train/test/val). If tier is None, then it is assumed that the the tier-key is not present.
        device (str, optional): device to compute on. Defaults to "cpu".
        as_dict (bool, optional): whether to return the results per data_key. Defaults to False.
        per_neuron (bool, optional): whether to return the results per neuron or averaged across neurons. Defaults to True.
        masking (bool,optional): If True it computes correlation only on first half of neurons (second half is seen by Encoder)
        forward_prior (bool,optional): If True latents are sampled from prior and marginalized
        forward_recursive (bool,optional): If True latents are sampled are copmuted recursivley
        repeats (int,optional): n_samples*repeats is the number of samples drawn from prior in total
        n_samples (int,optional): n_samples is the number of samples drawn in each repeat
        flow (boolean): If true applies flow to target responses before computing correlation
        neuron_mask (list): If given, masks the neurons with indices from the list plus first quarter of neurons and evaluates on first quarter,
        otherwise first half is masked and model is evaluated on first half.
        cell_coordinates (dict): contains dict of tensors of the neurons brain position for each mouse
    Returns:
        dict or np.ndarray: contains the correlation values.
    """
    correlations = {}
    dl = dataloaders[tier] if tier is not None else dataloaders
    for k, v in dl.items():
        target, output = model_predictions(
            dataloader=v,
            model=model,
            data_key=k,
            device=device,
            deeplake_ds=deeplake_ds,
            prior=forward_prior,
            recursive=forward_recursive,
            repeats=repeats,
            n_samples=n_samples,
            flow=flow,
            dropout_prob=dropout_prob,
            cell_coordinates=cell_coordinates,
            behavioral_modulation=behavioral_modulation,
        )
        target = np.concatenate(target, axis=1).T
        output = np.concatenate(output, axis=1).T
        if masking:
            number_neurons = target.shape[1]
            if dropout_prob:
                target = target[:, : (number_neurons // 4)]
                output = output[:, : (number_neurons // 4)]
            else:
                target = target[:, : (number_neurons // 2)]
                output = output[:, : (number_neurons // 2)]

        correlations[k] = corr(target, output, axis=0)

        if np.any(np.isnan(correlations[k])):
            warnings.warn(
                "{}% NaNs , NaNs will be set to Zero.".format(
                    np.isnan(correlations[k]).mean() * 100
                )
            )
        correlations[k][np.isnan(correlations[k])] = 0

    if not as_dict:
        correlations = (
            np.hstack([v for v in correlations.values()])
            if per_neuron
            else np.mean(np.hstack([v for v in correlations.values()]))
        )
    return correlations


def get_poisson_loss(
    model,
    dataloaders,
    device="cpu",
    as_dict=False,
    avg=False,
    per_neuron=True,
    eps=1e-12,
):
    poisson_loss = {}
    for k, v in dataloaders.items():
        target, output = model_predictions(
            dataloader=v, model=model, data_key=k, device=device
        )
        loss = output - target * np.log(output + eps)
        poisson_loss[k] = np.mean(loss, axis=0) if avg else np.sum(loss, axis=0)
    if as_dict:
        return poisson_loss
    else:
        if per_neuron:
            return np.hstack([v for v in poisson_loss.values()])
        else:
            return (
                np.mean(np.hstack([v for v in poisson_loss.values()]))
                if avg
                else np.sum(np.hstack([v for v in poisson_loss.values()]))
            )
