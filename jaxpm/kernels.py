import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
from jax.lib.xla_client import FftType
from jax.sharding import PartitionSpec as P
from jaxdecomp import fftfreq3d, get_output_specs

from jaxpm.distributed import autoshmap


def fftk(k_array):
    """
    Generate Fourier transform wave numbers for a given mesh.

    Args:
        nc (int): Shape of the mesh grid.

    Returns:
        list: List of wave number arrays for each dimension in
        the order [kx, ky, kz].
    """
    kx, ky, kz = fftfreq3d(k_array)
    # to the order of dimensions in the transposed FFT
    return kx, ky, kz


def interpolate_power_spectrum(input, k, pk, sharding=None):

    pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape(-1), k, pk
                                                  ).reshape(x.shape)

    gpu_mesh = sharding.mesh if sharding is not None else None
    specs = sharding.spec if sharding is not None else P()
    out_specs = P(*get_output_specs(
        FftType.FFT, specs, mesh=gpu_mesh)) if gpu_mesh is not None else P()

    return autoshmap(pk_fn,
                     gpu_mesh=gpu_mesh,
                     in_specs=out_specs,
                     out_specs=out_specs)(input)


def gradient_kernel(kvec, direction, order=1):
    """
    Computes the gradient kernel in the requested direction
    
    Parameters
    -----------
    kvec: list
        List of wave-vectors in Fourier space
    direction: int
        Index of the direction in which to take the gradient

    Returns
    --------
    wts: array
        Complex kernel values
    """
    if order == 0:
        wts = 1j * kvec[direction]
        wts = jnp.squeeze(wts)
        wts[len(wts) // 2] = 0
        wts = wts.reshape(kvec[direction].shape)
        return wts
    else:
        w = kvec[direction]
        a = 1 / 6.0 * (8 * jnp.sin(w) - jnp.sin(2 * w))
        wts = a * 1j
        return wts


def invlaplace_kernel(kvec):
    """
    Compute the inverse Laplace kernel

    Parameters
    -----------
    kvec: list
        List of wave-vectors

    Returns
    --------
    wts: array
        Complex kernel values
    """
    kk = sum(ki**2 for ki in kvec)
    kk_nozeros = jnp.where(kk==0, 1, kk) 
    return - jnp.where(kk==0, 0, 1 / kk_nozeros)


def longrange_kernel(kvec, r_split):
    """
    Computes a long range kernel

    Parameters
    -----------
    kvec: list
        List of wave-vectors
    r_split: float
        Splitting radius
        
    Returns
    --------
    wts: array
        Complex kernel values
    
    TODO: @modichirag add documentation
    """
    if r_split != 0:
        kk = sum(ki**2 for ki in kvec)
        return np.exp(-kk * r_split**2)
    else:
        return 1.


def cic_compensation(kvec):
    """
    Computes cic compensation kernel.
    Adapted from https://github.com/bccp/nbodykit/blob/a387cf429d8cb4a07bb19e3b4325ffdf279a131e/nbodykit/source/mesh/catalog.py#L499
    Itself based on equation 18 (with p=2) of
          [Jing et al 2005](https://arxiv.org/abs/astro-ph/0409240)

    Parameters:
    -----------
    kvec: list
        List of wave-vectors
        
    Returns:
    --------
    wts: array
        Complex kernel values
    """
    kwts = [np.sinc(kvec[i] / (2 * np.pi)) for i in range(3)]
    wts = (kwts[0] * kwts[1] * kwts[2])**(-2)
    return wts


def PGD_kernel(kvec, kl, ks):
    """
    Computes the PGD kernel

    Parameters:
    -----------
    kvec: list
        List of wave-vectors
    kl: float
        Initial long range scale parameter
    ks: float
        Initial dhort range scale parameter

    Returns:
    --------
    v: array
        Complex kernel values
    """
    kk = sum(ki**2 for ki in kvec)
    kl2 = kl**2
    ks4 = ks**4
    mask = (kk == 0).nonzero()
    kk[mask] = 1
    v = jnp.exp(-kl2 / kk) * jnp.exp(-kk**2 / ks4)
    imask = (~(kk == 0)).astype(int)
    v *= imask
    return v
