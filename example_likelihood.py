"""
Simple example script for computing likelihood trajectories.

This script demonstrates how to:
1. Load a pretrained model
2. Prepare sample data
3. Compute likelihood (bits/dim) for the samples
"""

import torch
import numpy as np
from models import utils as mutils
import sde_lib
import likelihood as likelihood_lib


def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale from [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def compute_likelihood(config, checkpoint_path, data):
  """
  Compute likelihood for a batch of data samples.

  Args:
    config: Configuration object (from configs/)
    checkpoint_path: Path to pretrained model checkpoint
    data: PyTorch tensor of shape [batch_size, channels, height, width]
          Values should be in [0, 1]

  Returns:
    bpd: Log-likelihood in bits/dim for each sample
    z: Latent codes
    nfe: Number of function evaluations
  """

  # Initialize SDE
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min,
                        beta_max=config.model.beta_max,
                        N=config.model.num_scales)
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min,
                           beta_max=config.model.beta_max,
                           N=config.model.num_scales)
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min,
                        sigma_max=config.model.sigma_max,
                        N=config.model.num_scales)
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Load model
  score_model = mutils.create_model(config)
  ckpt = torch.load(checkpoint_path, map_location=config.device)
  score_model.load_state_dict(ckpt)
  score_model.eval().to(config.device)

  # Prepare data
  scaler = get_data_scaler(config)
  inverse_scaler = get_data_inverse_scaler(config)
  data = data.to(config.device)
  data = scaler(data)

  # Create likelihood function
  likelihood_fn = likelihood_lib.get_likelihood_fn(
    sde, inverse_scaler,
    hutchinson_type='Rademacher',
    rtol=1e-5,
    atol=1e-5,
    method='RK45',
    eps=1e-5
  )

  # Compute likelihood
  bpd, z, nfe = likelihood_fn(score_model, data)

  return bpd, z, nfe


if __name__ == '__main__':
  # Example usage
  from configs.subvp import cifar10_ddpmpp_continuous

  # Load config
  config = cifar10_ddpmpp_continuous.get_config()

  # Create random sample data (replace with real data)
  batch_size = 4
  data = torch.rand(batch_size, 3, 32, 32)  # CIFAR-10 sized images

  # Path to your pretrained checkpoint
  checkpoint_path = 'path/to/checkpoint.pth'

  # Compute likelihood
  # bpd, z, nfe = compute_likelihood(config, checkpoint_path, data)
  # print(f"Bits/dim: {bpd}")
  # print(f"Number of function evaluations: {nfe}")

  print("Example setup complete!")
  print(f"To use: provide a checkpoint path and call compute_likelihood()")
  print(f"Data shape: {data.shape}")
  print(f"Config SDE type: {config.training.sde}")
