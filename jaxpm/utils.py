from functools import partial

import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm
from scipy.special import legendre

from jaxpm.kernels import compensation_kernel, gridding_shotnoise_kernel

__all__ = [
    'power_spectrum', 'transfer', 'coherence', 'pktranscoh',
    'cross_correlation_coefficients', 'gaussian_smoothing'
]


def _initialize_pk(mesh_shape, box_shape, kedges, los):
    """
    Parameters
    ----------
    mesh_shape : tuple of int
        Shape of the mesh grid.
    box_shape : tuple of float
        Physical dimensions of the box.
    kedges : None, int, float, or list
        If None, set dk to twice the minimum.
        If int, specifies number of edges.
        If float, specifies dk.
    los : array_like
        Line-of-sight vector.

    Returns
    -------
    dig : ndarray
        Indices of the bins to which each value in input array belongs.
    kcount : ndarray
        Count of values in each bin.
    kedges : ndarray
        Edges of the bins.
    mumesh : ndarray
        Mu values for the mesh grid.
    """
    kmax = np.pi * np.min(mesh_shape / box_shape)  # = knyquist

    if isinstance(kedges, None | int | float):
        if kedges is None:
            dk = 2 * np.pi / np.min(
                box_shape) * 2  # twice the minimum wavenumber
        if isinstance(kedges, int):
            dk = kmax / (kedges + 1)  # final number of bins will be kedges-1
        elif isinstance(kedges, float):
            dk = kedges
        kedges = np.arange(dk, kmax, dk) + dk / 2  # from dk/2 to kmax-dk/2

    kshapes = np.eye(len(mesh_shape), dtype=np.int32) * -2 + 1
    kvec = [(2 * np.pi * m / l) * np.fft.fftfreq(m).reshape(kshape)
            for m, l, kshape in zip(mesh_shape, box_shape, kshapes)]
    # NumPy (not jnp): kvec is static, so kmesh/dig/kcount/kavg/mumesh are all
    # compile-time constants. Using jnp.sqrt here would make kmesh a tracer
    kmesh = np.sqrt(sum(ki**2 for ki in kvec))

    dig = np.digitize(kmesh.reshape(-1), kedges)
    kcount = np.bincount(dig, minlength=len(kedges) + 1)

    # Central value of each bin
    # kavg = (kedges[1:] + kedges[:-1]) / 2
    kavg = np.bincount(
        dig, weights=kmesh.reshape(-1), minlength=len(kedges) + 1) / kcount
    kavg = kavg[1:-1]

    if los is None:
        mumesh = 1.
    else:
        mumesh = sum(ki * losi for ki, losi in zip(kvec, los))
        kmesh_nozeros = np.where(kmesh == 0, 1, kmesh)
        mumesh = np.where(kmesh == 0, 0, mumesh / kmesh_nozeros)

    return dig, kcount, kavg, mumesh


def power_spectrum(mesh,
                   mesh2=None,
                   box_shape=None,
                   kedges: int | float | list = None,
                   multipoles=0,
                   los=[0., 0., 1.],
                   compensate_order=None,
                   shotnoise=None):
    """
    Compute the auto and cross spectrum of 3D fields, with multipoles.

    Optional grid corrections (applied per-mode on the 3D field before binning,
    since both are anisotropic):

    compensate_order : int, str, or None
        If set, deconvolve the mass-assignment window: divide the power by
        ``W(k)**2`` with ``W = compensation_kernel(k, compensate_order)**(-1)``
        (i.e. multiply ``|delta_k|**2`` by ``compensation_kernel**2``).
    shotnoise : (order, nbar) or None
        If set, subtract the aliased shot noise ``(1 / nbar) * C_order(k)``
        before deconvolving (auto spectrum only). ``nbar`` is the mean number
        density in the same units as ``box_shape`` (so ``nbar = N / box.prod()``;
        for one particle per cell this is ``mesh.prod() / box.prod()``).
    """
    # Initialize
    mesh_shape = np.array(mesh.shape)
    if box_shape is None:
        box_shape = mesh_shape
    else:
        box_shape = np.asarray(box_shape)

    if multipoles == 0:
        los = None
    else:
        los = np.asarray(los)
        los = los / np.linalg.norm(los)
    poles = np.atleast_1d(multipoles)
    dig, kcount, kavg, mumesh = _initialize_pk(mesh_shape, box_shape, kedges,
                                               los)
    n_bins = len(kavg) + 2

    # FFTs
    meshk = jnp.fft.fftn(mesh, norm='ortho')
    if mesh2 is None:
        mmk = meshk.real**2 + meshk.imag**2
    else:
        mmk = meshk * jnp.fft.fftn(mesh2, norm='ortho').conj()

    # Optional grid corrections, applied per-mode on the 3D power before binning.
    # Both the window and the aliased shot noise are anisotropic, so they cannot
    # be applied to the already-binned (isotropic) P(k). The dealiasing kernel
    # lives in jaxpm.kernels and is reused here (master equation:
    #   P_true = (P_meas - (1/nbar) C_order) / W**2).
    if compensate_order is not None or shotnoise is not None:
        kshapes = np.eye(len(mesh_shape), dtype=np.int32) * -2 + 1
        kvec_cell = [
            2 * np.pi * np.fft.fftfreq(m).reshape(kshape)
            for m, kshape in zip(mesh_shape, kshapes)
        ]
        # cell-units shot noise in the same (mmk) units as |delta_k|**2:
        # physical shot is C/nbar, and mmk = P_phys / (box/mesh).prod().
        scale = (box_shape / mesh_shape).prod()
        if shotnoise is not None and mesh2 is None:
            sn_order, nbar = shotnoise
            mmk = mmk - gridding_shotnoise_kernel(kvec_cell,
                                                  sn_order) / (nbar * scale)
        if compensate_order is not None:
            mmk = mmk * compensation_kernel(kvec_cell, compensate_order)**2

    # Sum powers
    pk = jnp.empty((len(poles), n_bins))
    for i_ell, ell in enumerate(poles):
        weights = (mmk * (2 * ell + 1) * legendre(ell)(mumesh)).reshape(-1)
        if mesh2 is None:
            psum = jnp.bincount(dig, weights=weights, length=n_bins)
        else:  # XXX: bincount is really slow with complex numbers
            psum_real = jnp.bincount(dig, weights=weights.real, length=n_bins)
            psum_imag = jnp.bincount(dig, weights=weights.imag, length=n_bins)
            psum = (psum_real**2 + psum_imag**2)**.5
        pk = pk.at[i_ell].set(psum)

    # Normalization and conversion from cell units to [Mpc/h]^3
    pk = (pk / kcount)[:, 1:-1] * (box_shape / mesh_shape).prod()

    # pk = jnp.concatenate([kavg[None], pk])
    if np.ndim(multipoles) == 0:
        return kavg, pk[0]
    else:
        return kavg, pk


def transfer(mesh0, mesh1, box_shape, kedges: int | float | list = None):
    pk_fn = partial(power_spectrum, box_shape=box_shape, kedges=kedges)
    ks, pk0 = pk_fn(mesh0)
    ks, pk1 = pk_fn(mesh1)
    return ks, (pk1 / pk0)**.5


def coherence(mesh0, mesh1, box_shape, kedges: int | float | list = None):
    pk_fn = partial(power_spectrum, box_shape=box_shape, kedges=kedges)
    ks, pk01 = pk_fn(mesh0, mesh1)
    ks, pk0 = pk_fn(mesh0)
    ks, pk1 = pk_fn(mesh1)
    return ks, pk01 / (pk0 * pk1)**.5


def pktranscoh(mesh0, mesh1, box_shape, kedges: int | float | list = None):
    pk_fn = partial(power_spectrum, box_shape=box_shape, kedges=kedges)
    ks, pk01 = pk_fn(mesh0, mesh1)
    ks, pk0 = pk_fn(mesh0)
    ks, pk1 = pk_fn(mesh1)
    return ks, pk0, pk1, (pk1 / pk0)**.5, pk01 / (pk0 * pk1)**.5


def cross_correlation_coefficients(field_a,
                                   field_b,
                                   kmin=5,
                                   dk=0.5,
                                   boxsize=False):
    """
    Calculate the cross correlation coefficients given two real space field

    Args:

        field_a: real valued field
        field_b: real valued field
        kmin: minimum k-value for binned powerspectra
        dk: differential in each kbin
        boxsize: length of each boxlength (can be strangly shaped?)

    Returns:

        kbins: the central value of the bins for plotting
        P / norm: normalized cross correlation coefficient between two field a and b

  """
    shape = field_a.shape
    nx, ny, nz = shape

    #initialze values related to powerspectra (mode bins and weights)
    dig, Nsum, xsum, W, k, kedges = _initialize_pk(shape, boxsize, kmin, dk)

    #fast fourier transform
    fft_image_a = jnp.fft.fftn(field_a)
    fft_image_b = jnp.fft.fftn(field_b)

    #absolute value of fast fourier transform
    pk = fft_image_a * jnp.conj(fft_image_b)

    #calculating powerspectra
    real = jnp.real(pk).reshape([-1])
    imag = jnp.imag(pk).reshape([-1])

    Psum = jnp.bincount(dig, weights=(W.flatten() * imag),
                        length=xsum.size) * 1j
    Psum += jnp.bincount(dig, weights=(W.flatten() * real), length=xsum.size)

    P = ((Psum / Nsum)[1:-1] * boxsize.prod()).astype('float32')

    #normalization for powerspectra
    norm = np.prod(np.array(shape[:])).astype('float32')**2

    #find central values of each bin
    kbins = kedges[:-1] + (kedges[1:] - kedges[:-1]) / 2

    return kbins, P / norm


def gaussian_smoothing(im, sigma):
    """
  im: 2d image
  sigma: smoothing scale in px
  """
    # Compute k vector
    kvec = jnp.stack(jnp.meshgrid(jnp.fft.fftfreq(im.shape[0]),
                                  jnp.fft.fftfreq(im.shape[1])),
                     axis=-1)
    k = jnp.linalg.norm(kvec, axis=-1)
    # We compute the value of the filter at frequency k
    filter = norm.pdf(k, 0, 1. / (2. * np.pi * sigma))
    filter /= filter[0, 0]

    return jnp.fft.ifft2(jnp.fft.fft2(im) * filter).real
