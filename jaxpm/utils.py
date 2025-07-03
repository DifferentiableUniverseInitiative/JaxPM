from functools import partial

import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm
from scipy.special import legendre

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
    kmesh = jnp.sqrt(sum(ki**2 for ki in kvec))

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
                   los=[0., 0., 1.]):
    """
    Compute the auto and cross spectrum of 3D fields, with multipoles.
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


def flat_sky_power_spectrum(kappa_map, angle_deg, kappa_map2=None, ells=None):
    """
    Compute flat-sky angular power spectrum from 2D convergence maps.

    This function computes auto-power spectra (when kappa_map2=None) or
    cross-power spectra (when kappa_map2 is provided) using the flat-sky
    approximation where ℓ = k. The implementation matches lenstools for
    accurate cosmological analysis.

    Parameters
    ----------
    kappa_map : array_like, shape (N, N)
        2D convergence map or other projected field
    angle_deg : float
        Angular size of the map in degrees
    kappa_map2 : array_like, shape (N, N), optional
        Second map for cross-power spectrum. If None, computes auto-power.
    ells : array_like, optional
        Multipole bins for output. If None, uses logarithmic binning from
        ℓ_min = max(2π/θ, 10) to ℓ_max = π/pixel_scale with 20 bins.

    Returns
    -------
    ell_centers : array
        Centers of multipole bins
    cl : array
        Angular power spectrum C_ℓ (auto or cross)

    Notes
    -----
    The normalization follows the convention:
    C_ℓ = |FFT[κ]|² × (θ_rad²) / (N_pix⁴)

    This matches the lenstools.ConvergenceMap.powerSpectrum() implementation
    and is suitable for weak lensing and other flat-sky analyses.

    For cross-power spectra, computes:
    C_ℓ^{AB} = Re[FFT[A] × FFT[B]*] × (θ_rad²) / (N_pix⁴)

    Examples
    --------
    Auto-power spectrum:

    >>> ell, cl = flat_sky_power_spectrum(kappa_map, angle_deg=10.0)

    Cross-power spectrum:

    >>> ell, cl_cross = flat_sky_power_spectrum(map1, 10.0, map2)

    Custom ℓ binning:

    >>> ells = jnp.logspace(jnp.log10(50), jnp.log10(1000), 15)
    >>> ell, cl = flat_sky_power_spectrum(kappa_map, 10.0, ells=ells)
    """
    kappa_map = jnp.asarray(kappa_map)
    if kappa_map.ndim != 2 or kappa_map.shape[0] != kappa_map.shape[1]:
        raise ValueError("kappa_map must be a 2D square array")

    npix = kappa_map.shape[0]

    # Convert angle to radians
    angle_rad = angle_deg * jnp.pi / 180.0

    # Pixel scale in radians
    pixel_scale = angle_rad / npix

    # Remove mean (removes monopole)
    kappa_centered = kappa_map - jnp.mean(kappa_map)

    # 2D FFT
    kappa_fft = jnp.fft.fft2(kappa_centered)

    # Handle cross-power spectrum
    if kappa_map2 is not None:
        kappa_map2 = jnp.asarray(kappa_map2)
        if kappa_map2.shape != kappa_map.shape:
            raise ValueError("kappa_map2 must have same shape as kappa_map")

        kappa2_centered = kappa_map2 - jnp.mean(kappa_map2)
        kappa2_fft = jnp.fft.fft2(kappa2_centered)

        # Cross-power: Re[A × B*]
        power_2d = jnp.real(kappa_fft * jnp.conj(kappa2_fft))
    else:
        # Auto-power: |A|²
        power_2d = jnp.abs(kappa_fft)**2

    # Apply normalization: matches lenstools convention
    # Use float64 to avoid overflow for large npix
    normalization = (angle_rad**2) / jnp.float64(npix)**4
    power_2d = power_2d * normalization

    # Create wavenumber grids
    kx = jnp.fft.fftfreq(npix, pixel_scale) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(npix, pixel_scale) * 2 * jnp.pi
    kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing='ij')

    # Total wavenumber magnitude
    k_grid = jnp.sqrt(kx_grid**2 + ky_grid**2)

    # In flat sky approximation: ℓ = k
    ell_grid = k_grid

    # Default ell binning if not provided
    if ells is None:
        ell_min = max(2 * jnp.pi / angle_rad, 10)  # Fundamental mode or ℓ=10
        ell_max = jnp.pi / pixel_scale  # Nyquist limit
        ells = jnp.logspace(jnp.log10(float(ell_min)),
                            jnp.log10(float(ell_max)), 20)
    else:
        ells = jnp.asarray(ells)

    ell_centers = 0.5 * (ells[1:] + ells[:-1])

    # Bin the power spectrum
    cl_binned = []

    for i in range(len(ells) - 1):
        # Mask for this ell bin
        mask = (ell_grid >= ells[i]) & (ell_grid < ells[i + 1])

        n_modes = jnp.sum(mask)
        if n_modes > 0:
            # Average power in this bin
            power_sum = jnp.sum(power_2d * mask)
            cl_value = power_sum / n_modes
            cl_binned.append(cl_value)
        else:
            cl_binned.append(jnp.nan)

    cl_binned = jnp.array(cl_binned)

    # Remove NaN values
    valid_mask = ~jnp.isnan(cl_binned)
    ell_centers = ell_centers[valid_mask]
    cl_binned = cl_binned[valid_mask]

    return ell_centers, cl_binned
