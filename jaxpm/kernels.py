import jax
from jax.experimental.maps import xmap
import numpy as np
import jax.numpy as jnp
from functools import partial
import jaxdecomp

def fftk(shape, symmetric=False, dtype=np.float32, sharding_info=None):
    """ Return k_vector given a shape (nc, nc, nc)
    """
    k = []

    if sharding_info is not None:
        nx = sharding_info.pdims[1]
        ny = sharding_info.pdims[0]
        # nx = sharding_info[0].Get_size()
        # ix = sharding_info[0].Get_rank()
        # ny = sharding_info[1].Get_size()
        # iy = sharding_info[1].Get_rank()
        ix = sharding_info.rank % nx
        iy = 0
        shape = sharding_info.global_shape

    for d in range(len(shape)):
        kd = jnp.fft.fftfreq(shape[d])
        kd *= 2 * jnp.pi

        if symmetric and d == len(shape) - 1:
            kd = kd[:shape[d] // 2 + 1]

        if (sharding_info is not None) and d == 0:
            kd = kd.reshape([nx, -1])[ix]

        if (sharding_info is not None) and d == 1:
            kd = kd.reshape([ny, -1])[iy]

        k.append(kd.astype(dtype))
    return k


@jax.jit
def apply_gradient_laplace(kfield, kx, ky, kz):
    kk = (kx**2 + ky**2 + kz**2)
    kernel = jnp.where(kk == 0, 1., 1./kk)
    return jnp.stack([kfield * kernel * 1j * 1 / 6.0 * (8 * jnp.sin(kz) - jnp.sin(2 * kz)),
                      kfield * kernel * 1j * 1 / 6.0 * (8 * jnp.sin(kx) - jnp.sin(2 * kx)),
                      kfield * kernel * 1j * 1 / 6.0 * (8 * jnp.sin(ky) - jnp.sin(2 * ky))], axis=-1)


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
