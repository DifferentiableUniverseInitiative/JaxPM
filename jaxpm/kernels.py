from functools import partial

import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
from jax._src import mesh as mesh_lib
from jax.sharding import PartitionSpec as P

from jaxpm.distributed import autoshmap
from enum import Enum

class PencilType(Enum):
  NO_DECOMP = 0
  SLAB_XY = 1
  SLAB_YZ = 2
  PENCILS = 3

def get_pencil_type():
  mesh = mesh_lib.thread_resources.env.physical_mesh
  if mesh.empty:
    pdims = None
  else:
    pdims = mesh.devices.shape[::-1]

  if pdims == (1, 1) or pdims == None:
    return PencilType.NO_DECOMP
  elif pdims[0] == 1:
    return PencilType.SLAB_XY
  elif pdims[1] == 1:
    return PencilType.SLAB_YZ
  else:
    return PencilType.PENCILS

def fftk(shape, dtype=np.float32):
    """
    Generate Fourier transform wave numbers for a given mesh.

    Args:
        nc (int): Shape of the mesh grid.

    Returns:
        list: List of wave number arrays for each dimension in
        the order [kx, ky, kz].
  """
    kx, ky, kz = [jnp.fft.fftfreq(s, dtype=dtype) * 2 * np.pi for s in shape]

    @partial(autoshmap,
             in_specs=(P('x'), P('y'), P(None)),
             out_specs=(P('x'), P(None, 'y'), P(None)),in_fourrier_space=True)
    def get_kvec(ky, kz, kx):
        return (ky.reshape([-1, 1, 1]),
                kz.reshape([1, -1, 1]),
                kx.reshape([1, 1, -1])) # yapf: disable

    pencil_type = get_pencil_type() 
    # YZ returns Y pencil
    # XY and pencils returns a Z pencil
    # NO_DECOMP returns a X pencil
    if pencil_type == PencilType.NO_DECOMP:
        kx, ky, kz = get_kvec(kx, ky, kz) # Z Y X ==> X pencil
    elif pencil_type == PencilType.SLAB_YZ:
        kz, kx, ky = get_kvec(kz, kx, ky) # X Z Y ==> Y pencil
    elif pencil_type == PencilType.SLAB_XY or pencil_type == PencilType.PENCILS:
        ky, kz, kx = get_kvec(ky, kz, kx) # Z X Y ==> Z pencil
    else:
        raise ValueError("Unknown pencil type")

    # to the order of dimensions in the transposed FFT
    return kx, ky, kz


def interpolate_power_spectrum(input, k, pk):

    pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape(-1), k, pk
                                                  ).reshape(x.shape)
    return autoshmap(pk_fn, in_specs=P('x', 'y'), out_specs=P('x', 'y'),in_fourrier_space=True)(input)


def gradient_kernel(kvec, direction, order=1):
    """
  Computes the gradient kernel in the requested direction
  Parameters:
  -----------
  kvec: array
    Array of k values in Fourier space
  direction: int
    Index of the direction in which to take the gradient
  Returns:
  --------
  wts: array
    Complex kernel
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


def laplace_kernel(kvec):
    """
  Compute the Laplace kernel from a given K vector
  Parameters:
  -----------
  kvec: array
    Array of k values in Fourier space
  Returns:
  --------
  wts: array
    Complex kernel
  """
    kk = sum(ki**2 for ki in kvec)
    wts = jnp.where(kk == 0, 1., 1. / kk)
    return wts


def longrange_kernel(kvec, r_split):
    """
  Computes a long range kernel
  Parameters:
  -----------
  kvec: array
    Array of k values in Fourier space
  r_split: float
    TODO: @modichirag add documentation
  Returns:
  --------
  wts: array
    kernel
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
        `Jing et al 2005 <https://arxiv.org/abs/astro-ph/0409240>`_
  Args:
    kvec: array of k values in Fourier space
  Returns:
    v: array of kernel
  """
    kwts = [np.sinc(kvec[i] / (2 * np.pi)) for i in range(3)]
    wts = (kwts[0] * kwts[1] * kwts[2])**(-2)
    return wts


def PGD_kernel(kvec, kl, ks):
    """
  Computes the PGD kernel
  Parameters:
  -----------
  kvec: array
    Array of k values in Fourier space
  kl: float
    initial long range scale parameter
  ks: float
    initial dhort range scale parameter
  Returns:
  --------
  v: array
    kernel
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
