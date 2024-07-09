import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np

from jaxpm._src.spmd_config import pm_operators


def fftn(arr):
    return pm_operators.fftn(arr)


def ifftn(arr):
    return pm_operators.ifftn(arr)


def halo_exchange(arr):
    return pm_operators.halo_exchange(arr)


def slice_pad(arr, pad_width):
    return pm_operators.slice_pad(arr, pad_width)


def slice_unpad(arr, pad_width):
    return pm_operators.slice_unpad(arr, pad_width)


def normal(shape, key, dtype='float32'):
    return pm_operators.normal(shape, key, dtype)


def fftk(shape, symmetric=True, finite=False, dtype=np.float32):
    return pm_operators.fftk(shape, symmetric, finite, dtype)


def generate_initial_positions(shape):
    return pm_operators.generate_initial_positions(shape)


def interpolate_ic(kfield, kk, cosmo: jc.Cosmology, box_size):
    return pm_operators.interpolate_ic(kfield, kk, cosmo, box_size)
