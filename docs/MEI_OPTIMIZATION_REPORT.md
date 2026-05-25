# MEI Optimization Pipeline: Technical Report

## 1. Overview

Most Exciting Input (MEI) optimization finds the input stimulus that maximally activates a target neuron in a trained neural network model. Starting from random noise, the input is iteratively updated via gradient ascent on the neuron's predicted firing rate, subject to regularization constraints that encourage naturalistic image structure.

This report describes the MEI pipeline as implemented in the `mei` library and adapted for a 3D video sensorium model (`VideoFiringRateEncoder` with `Factorized3dCore`).

## 2. Core Algorithm

The optimization solves:

```
x* = argmax_x  f(x)
     subject to ||x||_p <= C,  x_min <= x_ij <= x_max
```

where `f(x)` is the model's predicted firing rate for a target neuron and the constraints prevent adversarial, out-of-distribution inputs.

### 2.1 Initialization

The MEI tensor is initialized from a standard normal distribution via `RandomNormal`:

```
x_0 ~ N(0, 1),  shape = (1, C, H, W)  or  (1, C, T, H, W) for video
```

The tensor is wrapped in an `Input` object that calls `requires_grad_()` to enable gradient tracking.

### 2.2 Optimization Step

Each iteration of `MEI.step()` executes the following operations in order:

```
For iteration t = 0, 1, 2, ..., N:

    1. TRANSFORM        x_t' = transform(x_t, t)
    2. EVALUATE          a_t = f(x_t')                    # forward pass
    3. BACKWARD          g_t = d(-a_t) / dx_t             # gradient ascent via negated loss
    4. PRECONDITION      g_t' = precondition(g_t, t)      # modify gradients
    5. OPTIMIZER STEP    x_{t+1} = x_t + lr * g_t'        # SGD update
    6. POSTPROCESS       x_{t+1} = postprocess(x_{t+1}, t) # constrain image
```

Key details:

- **Step 1 (Transform)**: Applied to a copy used for evaluation only; the stored MEI is not modified. The transform is lazily cached and computed once per iteration.
- **Step 3 (Backward)**: The loss is `(-evaluation + regularization_term)`. Negation converts gradient descent into gradient ascent.
- **Step 4 (Precondition)**: Modifies `tensor.grad` in-place before the optimizer reads it.
- **Step 6 (Postprocess)**: Operates on `tensor.data` directly, bypassing the computation graph.

### 2.3 Stopping and Tracking

The `optimize()` loop:

```python
while True:
    state = mei.step()                    # one optimization step
    stop, output = stopper(state)         # check termination
    tracker.track(state)                  # log objectives
    if stop: break
```

- **NumIterations stopper**: Terminates after a fixed number of iterations.
- **Tracker**: Logs the evaluation (predicted firing rate) at specified intervals via `EvaluationObjective`. Values are stored as `tracker.log['evaluation']['values']`.

## 3. Regularization Components

### 3.1 Vanilla Recipe

Uses only postprocessing for regularization:

| Component | Implementation | Effect |
|-----------|---------------|--------|
| Postprocessing | `PNormConstraintAndClip(norm=30, p=1, min=-1, max=1)` | Constrains L1 norm and clips pixel values |

This is minimal regularization. MEIs tend to have high-frequency features that exploit model sensitivities rather than reflecting naturalistic receptive field structure.

### 3.2 Paper Recipe (adapted from Walker & Sinz 2018)

Uses three regularization mechanisms:

| Component | Implementation | Effect |
|-----------|---------------|--------|
| Transform | `RandomJitter(amount=2)` | Shifts input ±2 pixels via `torch.roll` before each evaluation. Prevents pixel-aligned artifacts. |
| Precondition | `FourierPrecondition(alpha=0.6)` | Multiplies gradient FFT by `(f_x^2 + f_y^2)^{-\alpha}`. Amplifies low-frequency gradient components, suppresses high-frequency noise. |
| Postprocessing | `PNormConstraintAndClip(norm=30, p=1, min=-1, max=1)` | Same energy constraint as vanilla. |

### 3.3 Fourier Preconditioning Detail

The Fourier preconditioner is the primary smoothness mechanism. It operates on the gradient in frequency space:

```
G(f_x, f_y) = (f_x^2 + f_y^2)^{-alpha}
g'(x, y) = IFFT2[ FFT2[g(x, y)] * G(f_x, f_y) ]
```

The effect on the effective learning rate per spatial frequency:

| Frequency | alpha=0.1 | alpha=0.6 | alpha=1.2 |
|-----------|-----------|-----------|-----------|
| Low (f=0.05) | 1.6x | 5.7x | 33x |
| Mid (f=0.2) | 1.2x | 2.2x | 4.8x |
| High (f=0.5) | 1.0x (ref) | 1.0x (ref) | 1.0x (ref) |

At `alpha=0.6`, low-frequency components receive ~6x larger gradient steps than high-frequency components, producing visibly smoother MEIs while preserving the ability to form oriented features.

### 3.4 Design Decision: Why No Gaussian Blur

The original Walker & Sinz (2018) recipe uses a decaying Gaussian blur as postprocessing (sigma: 3.0 → 0.5 over iterations). This works for 2D CNN models with strong spatial gradients but fails for the 3D video model because:

1. **Energy drain**: Blur reduces the image's L2 norm each iteration. Without a compensating mechanism, the image collapses toward zero.
2. **Weak gradients**: The 3D video model's gradient signal is diluted by temporal expansion (`expand` over T frames) and averaging over T' output time steps, making it too weak to overcome the blur's destructive effect.
3. **Observed behavior**: Activation curves decrease monotonically and plateau at the model's baseline offset (~0.2), instead of increasing.

The Fourier preconditioning achieves the same goal (smoothness) by operating on the gradient side rather than the image side — it controls what structure gets added rather than destroying what's already there.

## 4. Adaptation for 3D Video Models

### 4.1 The ConstrainedOutputModel Problem

The `mei` library's `ConstrainedOutputModel` selects a target neuron via `output[:, constraint]`, indexing dimension 1. For 2D image models, output shape is `(batch, n_neurons)`, so this correctly selects a neuron. For the video model, output shape is `(batch, time, n_neurons)`, so dimension 1 is **time**, not neurons.

### 4.2 Custom Wrappers

Two wrapper classes replace `ConstrainedOutputModel`:

**VideoNeuronObjective** — optimizes a full video `(1, 1, T, H, W)`:
```python
def forward(self, x):
    output = model(x, data_key=data_key)   # (1, T', n_neurons)
    return output[:, :, neuron_idx].mean()  # select neuron, average over time
```

**StaticNeuronObjective** — optimizes a single frame `(1, 1, H, W)`:
```python
def forward(self, x):
    video = x.unsqueeze(2).expand(-1, -1, T, -1, -1)  # repeat frame T times
    output = model(video, data_key=data_key)            # (1, T', n_neurons)
    return output[:, :, neuron_idx].mean()              # select neuron, average over time
```

The static wrapper uses `expand` (not `repeat`) for memory efficiency — gradients correctly accumulate onto the single frame through the shared-memory view.

### 4.3 Video-Adapted Regularization

For video MEIs `(B, C, T, H, W)`, the 2D regularization callables are adapted:

- **FourierPreconditionVideo**: Applies 2D FFT per-frame (operates on last two dimensions). The `fft2`/`ifft2` functions broadcast naturally over leading dimensions.
- **RandomJitterVideo**: Rolls spatial dimensions 3 and 4 (instead of 2 and 3 for 2D).
- **PNormConstraintAndClip**: Works unchanged on arbitrary tensor shapes.

### 4.4 Temporal Considerations

The `Factorized3dCore` with `padding=False` reduces the temporal dimension. With kernels of size 11, 5, 5 across 3 layers, the temporal loss is 18 frames (input T=50 → output T'=32). The minimum viable input length is T=19 (yielding T'=1).

## 5. Pipeline Validation

A simulated complex cell (quadrature Gabor pair) serves as ground truth validation. The `ComplexCell` model computes:

```
response = sqrt( (x . g_even)^2 + (x . g_odd)^2 )
```

where `g_even` and `g_odd` are cosine- and sine-phase Gabor filters at the same orientation and spatial frequency. The MEI should recover a Gabor-like pattern at the correct orientation.

Both vanilla and paper recipes are validated on this model using the **identical** code path as the real model, ensuring the pipeline is correct before applying it to the trained network.

## 6. Summary

| Aspect | Vanilla | Paper |
|--------|---------|-------|
| Transform | None | RandomJitter (±2px) |
| Precondition | None | FourierPrecondition (alpha=0.6) |
| Postprocessing | PNormConstraintAndClip | PNormConstraintAndClip |
| Smoothness source | None (raw gradients) | Fourier lowpass on gradients |
| Typical activation | Higher | Lower (more constrained) |
| Visual quality | High-frequency artifacts | Smoother, more interpretable |

The paper recipe produces MEIs that better reflect the model's learned receptive field structure by suppressing optimization artifacts, at the cost of lower peak activation.
