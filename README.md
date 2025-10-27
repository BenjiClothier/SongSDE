# Score-Based Generative Modeling - Likelihood Computation (Simplified)

This is a **simplified version** of the [Score-Based Generative Modeling through Stochastic Differential Equations](https://openreview.net/forum?id=PxTIG12RRHS) codebase, focused solely on **computing likelihood trajectories** for samples.

Original paper by Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole (ICLR 2021).

---

## What does this simplified version do?

This codebase contains only the essential components needed to:
- Load pretrained score-based models
- Compute exact log-likelihoods (bits/dim) for data samples
- Calculate likelihood trajectories using the probability flow ODE

All training, sampling, and evaluation code has been removed to keep the codebase minimal and focused.

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**
- PyTorch >= 1.7.0
- NumPy
- SciPy (for ODE integration)
- ml-collections (for config management)

---

## Quick Start

See `example_likelihood.py` for a complete example:

```python
import torch
from configs.subvp import cifar10_ddpmpp_continuous
from example_likelihood import compute_likelihood

# Load config
config = cifar10_ddpmpp_continuous.get_config()

# Prepare your data (values in [0, 1])
data = torch.rand(4, 3, 32, 32)  # batch_size=4, RGB, 32x32

# Compute likelihood
checkpoint_path = 'path/to/checkpoint.pth'
bpd, z, nfe = compute_likelihood(config, checkpoint_path, data)

print(f"Bits/dim: {bpd}")
print(f"Latent codes: {z.shape}")
print(f"Function evaluations: {nfe}")
```

---

## Core Components

### Files Included

**Core likelihood computation:**
- `likelihood.py` - Main likelihood computation using probability flow ODE
- `sde_lib.py` - SDE implementations (VPSDE, subVPSDE, VESDE)
- `utils.py` - Checkpoint loading utilities
- `example_likelihood.py` - Simple usage example

**Model architectures:**
- `models/` - Score model implementations (NCSN++, DDPM++, etc.)
- `op/` - Custom operations used by models

**Configuration:**
- `configs/default_cifar10_configs.py` - Base configuration
- `configs/subvp/cifar10_ddpmpp_continuous.py` - Example config (subVPSDE excels at likelihood)

### How It Works

The likelihood computation uses:
1. **Probability Flow ODE**: Deterministic version of the reverse SDE
2. **Hutchinson Trace Estimator**: Efficiently computes divergence of the drift
3. **Black-box ODE Solver**: SciPy's `solve_ivp` integrates the ODE forward in time
4. **Prior Log-Probability**: Computes final likelihood from Gaussian prior

The trajectory starts at your data point and flows to the prior distribution, accumulating log-probability changes along the way.

---

## Pretrained Checkpoints

Download pretrained models from the [original repository's Google Drive](https://drive.google.com/drive/folders/1tFmF_uh57O6lx9ggtZT_5LdonVK2cV-e?usp=sharing).

**Recommended for likelihood computation:**
- `subvp/cifar10_ddpmpp_deep_continuous` - Best likelihood: **2.99 bits/dim** on CIFAR-10
- `subvp/cifar10_ddpmpp_continuous` - Good balance: **3.02 bits/dim**
- `vp/cifar10_ddpmpp_deep_continuous` - Alternative: **3.13 bits/dim**

The subVPSDE models generally achieve the best likelihood values.

---

## Configuration

Configs specify:
- **SDE type**: `vpsde`, `subvpsde`, or `vesde`
- **Model architecture**: Model parameters and architecture type
- **Data properties**: Image size, channels, centering, etc.

You can modify configs or create new ones in the `configs/` directory.

---

## API Reference

### `likelihood.get_likelihood_fn(sde, inverse_scaler, **kwargs)`

Creates a likelihood computation function.

**Arguments:**
- `sde`: SDE object from `sde_lib` (VPSDE, subVPSDE, or VESDE)
- `inverse_scaler`: Function to convert normalized data back to [0, 1]
- `hutchinson_type`: `'Rademacher'` or `'Gaussian'` (default: Rademacher)
- `rtol`: Relative tolerance for ODE solver (default: 1e-5)
- `atol`: Absolute tolerance for ODE solver (default: 1e-5)
- `method`: ODE solver method (default: 'RK45')
- `eps`: Integration end time for stability (default: 1e-5)

**Returns:**
Function `likelihood_fn(model, data)` that returns:
- `bpd`: Bits per dimension (batch_size,)
- `z`: Latent codes (same shape as data)
- `nfe`: Number of function evaluations (int)

---

## What Was Removed?

To keep this codebase minimal, the following were removed:
- Training code (`run_lib.py`, `losses.py`, `main.py`)
- Sampling methods (`sampling.py`)
- Data loading (`datasets.py`)
- Evaluation metrics (`evaluation.py` - FID, Inception Score)
- Controllable generation (`controllable_generation.py`)
- Demo notebooks and assets
- TensorFlow dependencies

If you need these features, please use the [original repository](https://github.com/yang-song/score_sde_pytorch).

---

## References

If you use this code, please cite the original paper:

```bibtex
@inproceedings{
  song2021scorebased,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Yang Song and Jascha Sohl-Dickstein and Diederik P Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=PxTIG12RRHS}
}
```

---

## License

Apache 2.0 (same as original repository)

---

## Original Repository

For the full implementation including training, sampling, and evaluation:
https://github.com/yang-song/score_sde_pytorch
