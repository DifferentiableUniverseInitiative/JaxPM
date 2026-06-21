import warnings

import jax.numpy as jnp
import numpy as np
from jax.lax import FftType
from jax.sharding import PartitionSpec as P
from jaxdecomp import fftfreq3d, get_fft_output_sharding

from jaxpm.distributed import autoshmap

# Mapping from mass-assignment scheme name to its order (number of cells the
# kernel spans per dimension). Shared with jaxpm.painting.
_ORDER = {'ngp': 1, 'cic': 2, 'tsc': 3, 'pcs': 4}


def resolve_order(order):
    """Normalise a painting order to its integer code (1-4).

    Accepts either a name ('NGP', 'CIC', 'TSC', 'PCS', case-insensitive) or an
    integer 1-4 (NGP=1, CIC=2, TSC=3, PCS=4).
    """
    if isinstance(order, str):
        try:
            return _ORDER[order.lower()]
        except KeyError:
            raise ValueError(
                f"Unknown painting order {order!r}; expected one of "
                f"{sorted(_ORDER)} or an integer 1-4") from None
    order = int(order)
    if order not in (1, 2, 3, 4):
        raise ValueError(f"Painting order must be in 1-4, got {order}")
    return order


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

    pk_fn = lambda x: jnp.interp(x, k, pk)

    gpu_mesh = sharding.mesh if sharding is not None else None
    specs = sharding.spec if sharding is not None else P()
    out_specs = get_fft_output_sharding(
        sharding).spec if sharding is not None else P()

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
        wts = wts.at[len(wts) // 2].set(0)
        wts = wts.reshape(kvec[direction].shape)
        return wts
    else:
        w = kvec[direction]
        a = 1 / 6.0 * (8 * jnp.sin(w) - jnp.sin(2 * w))
        wts = a * 1j
        return wts


def invlaplace_kernel(kvec, fd=False):
    """
    Compute the inverse Laplace kernel.

    cf. [Feng+2016](https://arxiv.org/pdf/1603.00476)

    Parameters
    -----------
    kvec: list
        List of wave-vectors
    fd: bool
        Finite difference kernel

    Returns
    --------
    wts: array
        Complex kernel values
    """
    if fd:
        kk = sum((ki * jnp.sinc(ki / (2 * jnp.pi)))**2 for ki in kvec)
    else:
        kk = sum(ki**2 for ki in kvec)
    kk_nozeros = jnp.where(kk == 0, 1, kk)
    return -jnp.where(kk == 0, 0, 1 / kk_nozeros)


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


def compensation_kernel(kvec, order):
    """Window-function (de)compensation kernel for a mass-assignment scheme.

    Painting a particle onto the grid convolves the field with the assignment
    window ``W(k) = prod_i sinc(k_i / 2)^order`` (``jnp.sinc(k/(2*pi)) =
    sin(k/2)/(k/2)``), which suppresses small scales. This returns the inverse
    window ``W(k)^(-order)``, i.e. multiplying a painted *field* by this kernel
    in Fourier space deconvolves it (removes one factor of the window).

    For a *power spectrum* the window enters squared, so square this kernel
    (see :func:`jaxpm.utils.power_spectrum`).

    Based on equation 18 of [Jing et al 2005](https://arxiv.org/abs/astro-ph/0409240),
    generalised from CIC (p=2) to arbitrary order.

    Parameters
    ----------
    kvec : list
        List of wave-vectors.
    order : int or str
        Assignment order: NGP=1, CIC=2, TSC=3, PCS=4 (name or integer).

    Returns
    -------
    wts : array
        Real kernel values ``(prod_i sinc(k_i/(2*pi)))**(-order)``.
    """
    order = resolve_order(order)
    kwts = [jnp.sinc(kvec[i] / (2 * np.pi)) for i in range(3)]
    return (kwts[0] * kwts[1] * kwts[2])**(-order)


def gridding_shotnoise_kernel(kvec, order):
    """Aliased shot-noise (dealiasing) kernel ``C_order(k)`` for a scheme.

    The discreteness (Poisson) shot noise of particles painted with a given
    assignment scheme is not flat but aliased: its expectation is
    ``(1 / nbar) * C_order(k)``, where ``C_order(k) = prod_i C_order(k_i)`` and
    each per-dimension factor is the closed-form sum of ``|W|^2`` over aliases
    (Jing et al 2005). To *dealias* a measured power spectrum, subtract
    ``(1 / nbar) * gridding_shotnoise_kernel(kvec, order)`` before deconvolving.

    This is a power-spectrum-level correction and is intentionally **not**
    applied inside :func:`jaxpm.painting.paint` (it cannot be expressed as a
    filter on a field); it is exposed here as a standalone kernel.

    With ``s2 = sin(k_i / 2)**2`` the per-dimension factors are:
      - NGP : 1
      - CIC : 1 - (2/3) s2
      - TSC : 1 - s2 + (2/15) s2**2
      - PCS : 1 - (4/3) s2 + (2/5) s2**2 - (4/315) s2**3

    Parameters
    ----------
    kvec : list
        List of wave-vectors.
    order : int or str
        Assignment order: NGP=1, CIC=2, TSC=3, PCS=4 (name or integer).

    Returns
    -------
    wts : array
        Dimensionless ``C_order(k)``; multiply by ``1 / nbar`` for the noise.
    """
    order = resolve_order(order)

    def per_dim(ki):
        s2 = jnp.sin(ki / 2.0)**2
        if order == 1:
            return jnp.ones_like(s2)
        elif order == 2:
            return 1. - 2. / 3. * s2
        elif order == 3:
            return 1. - s2 + 2. / 15. * s2**2
        else:  # order == 4
            return 1. - 4. / 3. * s2 + 2. / 5. * s2**2 - 4. / 315. * s2**3

    return per_dim(kvec[0]) * per_dim(kvec[1]) * per_dim(kvec[2])


def cic_compensation(kvec):
    """Deprecated: use :func:`compensation_kernel` with ``order='cic'`` instead."""
    warnings.warn(
        "cic_compensation is deprecated; use "
        "compensation_kernel(kvec, order) instead.",
        DeprecationWarning,
        stacklevel=2)
    return compensation_kernel(kvec, 2)


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
    kk_nozero = jnp.where(kk == 0, 1,
                          kk)  # avoid 0-division at k=0 (masked below)
    v = jnp.exp(-kl2 / kk_nozero) * jnp.exp(-kk_nozero**2 / ks4)
    imask = (~(kk == 0)).astype(int)
    v *= imask
    return v
