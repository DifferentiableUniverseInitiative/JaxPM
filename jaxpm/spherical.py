from functools import partial
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np

Array = jnp.ndarray


def _smoothing_sigma_rad(
    nside: int,
    kernel_width_arcmin: Optional[float] = None,
    kernel_width_pixels: Optional[float] = None,
    smoothing_interpretation: str = "fwhm",
):
    """Resolve a smoothing width to a Gaussian ``sigma`` in radians.

    The width may be given in arcminutes (``kernel_width_arcmin``) or in HEALPix
    pixels (``kernel_width_pixels`` -- a float, so ``0.5`` = half a pixel),
    the latter converted with the pixel angular scale ``jhp.nside2resol(nside)``.
    The value is interpreted via ``smoothing_interpretation`` ('fwhm', 'sigma',
    '2sigma'). If neither width is given, defaults to a half-pixel sigma
    (``resol / 2``), matching the historical RBF default.

    Shared by :func:`paint_particles_spherical_rbf_neighbor` and
    :func:`deconvolve_map` so the real-space painting kernel and the harmonic
    deconvolution beam use the *same* ``sigma`` (and therefore cancel).
    """
    if kernel_width_pixels is not None and kernel_width_arcmin is not None:
        raise ValueError(
            "Pass only one of kernel_width_pixels or kernel_width_arcmin.")

    # nside is a static argument in both callers, so the pixel scale is a concrete
    # host float. Using plain arithmetic (no jnp.asarray) keeps the result a Python
    # float for concrete widths -- so deconvolve_map can take float(sigma) under
    # jit -- while still tracing through a *traced* width in the painter.
    resol = float(jhp.nside2resol(nside))
    if kernel_width_pixels is not None:
        width_rad = kernel_width_pixels * resol
    elif kernel_width_arcmin is not None:
        width_rad = kernel_width_arcmin * (np.pi / 180.0) / 60.0
    else:
        # Default: half-pixel sigma (preserves the previous RBF default).
        return resol / 2.0

    if smoothing_interpretation == "fwhm":
        return width_rad / 2.355
    elif smoothing_interpretation == "2sigma":
        return width_rad / 2.0
    elif smoothing_interpretation == "sigma":
        return width_rad
    else:
        raise ValueError(
            "smoothing_interpretation must be one of 'fwhm', 'sigma', or '2sigma'"
        )


def _allocate_healpix_map(
        nside: int,
        dtype=jnp.float32,
        sharding: Optional[jax.sharding.Sharding] = None) -> Array:
    npix = jhp.nside2npix(nside)
    hp_map = jnp.zeros(npix, dtype=dtype)
    if sharding is not None:
        sharding_1d = jax.sharding.NamedSharding(
            sharding.mesh, jax.sharding.PartitionSpec(sharding.spec[0]))
        hp_map = jax.lax.with_sharding_constraint(hp_map, sharding_1d)
    return hp_map


@partial(jax.jit, static_argnames=("nside", "sharding"))
def paint_particles_spherical_ngp(
    positions: Array,
    nside: int,
    observer_position: Union[Array, jnp.ndarray],
    R_min: float,
    R_max: float,
    box_size: Union[float, Array, jnp.ndarray],
    mesh_shape: Tuple[int, int, int],
    weights: Optional[Array] = None,
    sharding: Optional[jax.sharding.Sharding] = None,
) -> Array:
    """
    Paint particles onto HEALPix spherical maps using Nearest Grid Point (NGP) scheme.

    This is a fast method that assigns particles to the nearest pixel.

    Parameters
    ----------
    positions : ndarray, shape (..., 3)
        Particle positions in simulation coordinates
    nside : int
        HEALPix nside parameter
    observer_position : ndarray, shape (3,)
        Observer position in comoving coordinates
    R_min, R_max : float
        Minimum and maximum comoving distance range to include
    box_size : float or array
        Size of the simulation box in physical units
    mesh_shape : tuple
        Shape of the simulation mesh (nx, ny, nz)
    weights : ndarray, optional
        Particle weights (default: uniform weights)
    sharding : jax.sharding.Sharding, optional
        Sharding information for distributed computation. If provided, the HEALPix map
        will be allocated with the specified sharding
    Returns
    -------
    healpix_map : ndarray
        HEALPix density map
    """
    # Convert particle positions to physical coordinates
    positions_phys = positions * jnp.array(box_size) / jnp.array(mesh_shape)

    # Compute relative positions from observer
    rel_positions = positions_phys - jnp.asarray(observer_position)

    # Comoving distance from observer
    r = jnp.linalg.norm(rel_positions, axis=-1)

    if weights is None:
        weights = jnp.ones_like(r)

    # Apply distance cuts - use masking instead of boolean indexing
    distance_mask = (r >= R_min) & (r <= R_max)

    # Apply mask to weights (original shape preserved)
    masked_weights = jnp.where(distance_mask, weights, 0.0)

    # Safe division to avoid division by zero
    r_safe = jnp.where(r > 1e-10, r, 1e-10)
    unit_vecs = rel_positions / r_safe[..., None]

    # Convert unit vectors to angles using jax_healpy (preserves batch shape)
    theta, phi = jhp.vec2ang(unit_vecs)

    # Convert to HEALPix pixel indices (preserves batch shape)
    pixels = jhp.ang2pix(nside, theta, phi)

    # Bin particles into HEALPix pixels (flatten here for bincount)
    npix = jhp.nside2npix(nside)
    healpix_map = _allocate_healpix_map(nside,
                                        dtype=masked_weights.dtype,
                                        sharding=sharding)
    healpix_map = healpix_map.at[pixels].add(masked_weights)

    # Calculate volume per pixel in spherical shell (exact shell volume)
    pixel_solid_angle = 4 * jnp.pi / npix  # steradians per pixel
    shell_volume_per_pixel = pixel_solid_angle * (R_max**3 - R_min**3) / 3.0

    # Convert particle counts to density (particles per unit volume)
    return healpix_map / shell_volume_per_pixel


@partial(jax.jit, static_argnames=("nside", "sharding"))
def paint_particles_spherical_bilinear(
    positions: Array,
    nside: int,
    observer_position: Union[Array, jnp.ndarray],
    R_min: float,
    R_max: float,
    box_size: Union[float, Array, jnp.ndarray],
    mesh_shape: Tuple[int, int, int],
    weights: Optional[Array] = None,
    sharding: Optional[jax.sharding.Sharding] = None,
) -> Array:
    """
    Paint particles onto HEALPix spherical maps using bilinear interpolation.

    This implements bilinear interpolation using jax_healpy.get_interp_weights()
    to distribute each particle's contribution among its 4 nearest pixels.

    Parameters
    ----------
    positions : ndarray, shape (..., 3)
        Particle positions in simulation coordinates
    nside : int
        HEALPix nside parameter
    observer_position : ndarray, shape (3,)
        Observer position in comoving coordinates
    R_min, R_max : float
        Minimum and maximum comoving distance range to include
    box_size : float or array
        Size of the simulation box in physical units
    mesh_shape : tuple
        Shape of the simulation mesh (nx, ny, nz)
    weights : ndarray, optional
        Particle weights (default: uniform weights)
    sharding : jax.sharding.Sharding, optional
        Sharding information for distributed computation. If provided, the HEALPix map
        will be allocated with the specified sharding

    Returns
    -------
    healpix_map : ndarray
        HEALPix density map
    """
    # Convert particle positions to physical coordinates
    positions_phys = positions * jnp.array(box_size) / jnp.array(mesh_shape)

    # Compute relative positions from observer
    rel_positions = positions_phys - jnp.asarray(observer_position)

    # Comoving distance from observer
    r = jnp.linalg.norm(rel_positions, axis=-1)

    if weights is None:
        weights = jnp.ones_like(r)

    # Apply distance cuts using masking (no boolean indexing)
    distance_mask = (r >= R_min) & (r <= R_max)

    # Apply mask to weights (original shape preserved)
    masked_weights = jnp.where(distance_mask, weights, 0.0)

    # Safe division to avoid division by zero
    r_safe = jnp.where(r > 1e-10, r, 1e-10)
    unit_vecs = rel_positions / r_safe[..., None]

    # Convert unit vectors to spherical coordinates (preserves batch shape)
    theta, phi = jhp.vec2ang(unit_vecs)

    # Get bilinear interpolation weights and pixel indices: (4, *batch)
    pixels, interp_weights = jhp.get_interp_weights(nside, theta, phi)

    # Initialize HEALPix map
    npix = jhp.nside2npix(nside)
    healpix_map = _allocate_healpix_map(nside,
                                        dtype=masked_weights.dtype,
                                        sharding=sharding)

    # interp_weights: (4, *batch), masked_weights: (*batch,) broadcasts to (4, *batch)
    contributions = interp_weights * masked_weights
    # Scatter contributions (flatten for at[].add — communication unavoidable)
    healpix_map = healpix_map.at[pixels].add(contributions)

    # Apply shell-volume normalization
    pixel_area = 4 * jnp.pi / npix
    shell_vol_per_pix = pixel_area * (R_max**3 - R_min**3) / 3
    return healpix_map / shell_vol_per_pix


@partial(jax.jit,
         static_argnames=("nside", "smoothing_interpretation", "sharding"))
def paint_particles_spherical_rbf_neighbor(
    positions: Array,
    nside: int,
    observer_position: Union[Array, jnp.ndarray],
    R_min: float,
    R_max: float,
    box_size: Union[float, Array, jnp.ndarray],
    mesh_shape: Tuple[int, int, int],
    weights: Optional[Array] = None,
    kernel_width_arcmin: Optional[float] = None,
    kernel_width_pixels: Optional[float] = None,
    smoothing_interpretation: str = "fwhm",
    sharding: Optional[jax.sharding.Sharding] = None,
) -> Array:
    """
    Paint particles onto HEALPix spherical maps using RBF with fixed neighbor stencil.

    This implements a Gaussian RBF kernel using a fixed stencil of 9 pixels
    (central pixel + 8 neighbors) per particle for predictable performance.

    Parameters
    ----------
    positions : ndarray, shape (..., 3)
        Particle positions in simulation coordinates
    nside : int
        HEALPix nside parameter
    observer_position : ndarray, shape (3,)
        Observer position in comoving (MAKE SURE THAT IT WORKS IF NOT IN THE CENTER)
    R_min, R_max : float
        Minimum and maximum comoving distance range to include
    box_size : float or array
        Size of the simulation box in physical units
    mesh_shape : tuple
        Shape of the simulation mesh (nx, ny, nz)
    weights : ndarray, optional
        Particle weights (default: uniform weights)
    kernel_width_arcmin : float, optional
        Width of the Gaussian smoothing kernel in arcminutes.
        Larger values → more smoothing (blurrier maps).
        Smaller values → less smoothing (sharper maps).
    kernel_width_pixels : float, optional
        Width of the Gaussian smoothing kernel in HEALPix pixels (a float, so
        ``0.5`` smooths by half a pixel). Converted with ``jhp.nside2resol``.
        Mutually exclusive with ``kernel_width_arcmin``. If both are None the
        kernel defaults to a half-pixel sigma.
    smoothing_interpretation : {"fwhm", "sigma", "2sigma"}
        Interpretation of the kernel width:
        - 'fwhm': the width is the full-width at half-maximum
        - 'sigma': the width is the standard deviation
        - '2sigma': the width is 2× the standard deviation
    sharding : jax.sharding.Sharding, optional
        Sharding information for distributed computation. If provided, the HEALPix map
        will be allocated with the specified sharding

    Returns
    -------
    healpix_map : ndarray
        HEALPix density map
    """
    sigma = _smoothing_sigma_rad(
        nside,
        kernel_width_arcmin=kernel_width_arcmin,
        kernel_width_pixels=kernel_width_pixels,
        smoothing_interpretation=smoothing_interpretation,
    )

    # Convert particle positions to physical coordinates
    positions_phys = positions * jnp.array(box_size) / jnp.array(mesh_shape)

    # Compute relative positions from observer
    rel_positions = positions_phys - jnp.asarray(observer_position)

    # Comoving distance from observer
    r = jnp.linalg.norm(rel_positions, axis=-1)

    if weights is None:
        weights = jnp.ones_like(r)

    # Apply distance cuts using masking (no boolean indexing)
    distance_mask = (r >= R_min) & (r <= R_max)

    # Apply mask to weights (original shape preserved)
    masked_weights = jnp.where(distance_mask, weights, 0.0)

    # Safe division to avoid division by zero
    r_safe = jnp.where(r > 1e-10, r, 1e-10)
    unit_vecs = rel_positions / r_safe[..., None]

    # Convert unit vectors to spherical coordinates (preserves batch shape)
    theta, phi = jhp.vec2ang(unit_vecs)

    # Get 9-pixel stencils: center + 8 neighbors -> (9, *batch)
    pix9 = jhp.get_all_neighbours(nside,
                                  theta,
                                  phi,
                                  nest=False,
                                  get_center=True)

    # Get unit vectors for all 9 pixels -> (9, *batch, 3)
    vecs = jhp.pix2vec(nside, pix9)
    # NaNs can arise for neighbors with index -1 (non-existent); keep vectors but mask later.
    vecs = jnp.nan_to_num(vecs)

    # Compute angular separations: (*batch, 3) broadcasts with (9, *batch, 3) -> sum over axis=-1 -> (9, *batch)
    dots = jnp.sum(unit_vecs * vecs, axis=-1)
    gamma = jnp.arccos(jnp.clip(dots, -1.0, 1.0))

    # Gaussian kernel weights
    kernel_weights = jnp.exp(-(gamma**2) /
                             (2 * sigma**2)) / (2 * jnp.pi * sigma**2)

    # Mask invalid neighbors (pix == -1) and renormalize per particle to conserve mass
    valid_mask = (pix9 != -1)
    kernel_weights_masked = jnp.where(valid_mask, kernel_weights, 0.0)
    weight_sum = jnp.sum(kernel_weights_masked, axis=0)  # (*batch,)
    # Safe normalization: if no valid neighbors, keep zeros
    norm_kernel = jnp.where(weight_sum[None, ...] > 0.0,
                            kernel_weights_masked / weight_sum[None, ...], 0.0)

    # Initialize HEALPix map size
    npix = jhp.nside2npix(nside)

    # Weight by particle weight; kernel sums to 1 per particle
    pixel_area = 4.0 * jnp.pi / npix
    # masked_weights: (*batch,) broadcasts with (9, *batch)
    contrib = norm_kernel * masked_weights

    # Scatter contributions into a map
    # Avoid negative indices by zeroing their contributions and redirecting index to 0
    valid = (pix9 != -1)
    idx_safe = jnp.where(valid, pix9, 0)
    val_safe = jnp.where(valid, contrib, 0.0)
    healpix_map = _allocate_healpix_map(nside,
                                        dtype=masked_weights.dtype,
                                        sharding=sharding)
    healpix_map = healpix_map.at[idx_safe].add(val_safe)

    # Apply shell-volume normalization
    shell_vol_per_pix = pixel_area * (R_max**3 - R_min**3) / 3
    return healpix_map / shell_vol_per_pix


@partial(jax.jit,
         static_argnames=("nside", "method", "ud_grade_order_in",
                          "ud_grade_order_out", "ud_grade_power",
                          "ud_grade_pess", "paint_nside",
                          "smoothing_interpretation", "sharding"))
def paint_particles_spherical(
    positions: Array,
    nside: int,
    observer_position: Union[Array, jnp.ndarray],
    R_min: float,
    R_max: float,
    box_size: Union[float, Array, jnp.ndarray],
    mesh_shape: Tuple[int, int, int],
    weights: Optional[Array] = None,
    method: str = "ngp",
    kernel_width_arcmin: Optional[float] = None,
    kernel_width_pixels: Optional[float] = None,
    smoothing_interpretation: str = "fwhm",
    # High-resolution painting option
    paint_nside: Optional[int] = None,
    ud_grade_power: float = 0.0,
    ud_grade_order_in: str = "RING",
    ud_grade_order_out: str = "RING",
    ud_grade_pess: bool = False,
    # Sharding infomration
    sharding: Optional[jax.sharding.Sharding] = None,
) -> Array:
    """
    High-level spherical painter: select method and optionally paint at higher resolution.

    This function dispatches to one of the three fast painting methods and optionally
    paints at higher resolution before downgrading to the target resolution.

    Parameters
    ----------
    positions : ndarray, shape (..., 3)
        Particle positions in simulation coordinates
    nside : int
        HEALPix nside parameter for final output
    observer_position : ndarray, shape (3,)
        Observer position in comoving coordinates
    R_min, R_max : float
        Minimum and maximum comoving distance range to include
    box_size : float or array
        Size of the simulation box in physical units
    mesh_shape : tuple
        Shape of the simulation mesh (nx, ny, nz)
    weights : ndarray, optional
        Particle weights (default: uniform weights)
    method (case-insensitive): str
        Painting method: 'ngp', 'bilinear', or 'rbf_neighbor'
    kernel_width_arcmin : float, optional
        Width of the Gaussian kernel in arcminutes for the RBF method.
        Larger values → more smoothing. Smaller values → less smoothing.
    kernel_width_pixels : float, optional
        Width of the Gaussian kernel in HEALPix pixels for the RBF method (a
        float, so ``0.5`` = half a pixel). Mutually exclusive with
        ``kernel_width_arcmin``.
    smoothing_interpretation : {"fwhm", "sigma", "2sigma"}
        Interpretation of the RBF kernel width:
        - 'fwhm': full-width at half-maximum
        - 'sigma': standard deviation
        - '2sigma': 2× standard deviation
    paint_nside : int, optional
        Internal resolution to paint at. If None, equals nside.
    ud_grade_power : float
        Power parameter for ud_grade
    ud_grade_order_in : str
        Input pixel ordering for ud_grade
    ud_grade_order_out : str
        Output pixel ordering for ud_grade
    ud_grade_pess : bool
        Pessimistic flag for ud_grade
    sharding : jax.sharding.Sharding, optional
        Sharding information for distributed computation. If provided, the HEALPix map
        will be allocated with the specified sharding but using only the first dimension.
        This means that most effective domain decomposition will be slab-based along the first dimension
        If sharding is provided, all the operations are guarenteed to use all the available devices
        and the output map will be sharded according to the provided sharding.

    Returns
    -------
    healpix_map : ndarray
        HEALPix density map at resolution nside
    """
    method_upper = method.strip().upper()

    # Determine internal nside for painting
    internal_nside = int(paint_nside) if paint_nside is not None else int(
        nside)

    if weights is None:
        # Alternative, shard perserving ones_like
        weights = positions[
            ..., 0] * 0.0 + 1.0  # shape (...,) with same sharding as positions

    # Select appropriate painter
    if method_upper == "NGP":
        map_hi = paint_particles_spherical_ngp(
            positions,
            internal_nside,
            observer_position,
            R_min,
            R_max,
            box_size,
            mesh_shape,
            weights=weights,
            sharding=sharding,
        )
    elif method_upper == "BILINEAR":
        map_hi = paint_particles_spherical_bilinear(
            positions,
            internal_nside,
            observer_position,
            R_min,
            R_max,
            box_size,
            mesh_shape,
            weights=weights,
            sharding=sharding,
        )
    elif method_upper == "RBF_NEIGHBOR":
        map_hi = paint_particles_spherical_rbf_neighbor(
            positions,
            internal_nside,
            observer_position,
            R_min,
            R_max,
            box_size,
            mesh_shape,
            weights=weights,
            kernel_width_arcmin=kernel_width_arcmin,
            kernel_width_pixels=kernel_width_pixels,
            smoothing_interpretation=smoothing_interpretation,
            sharding=sharding,
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose 'ngp', 'bilinear', or 'rbf_neighbor'."
        )

    # If internal resolution differs from target, up/down-grade
    if internal_nside != int(nside):
        return jhp.ud_grade(
            map_hi,
            int(nside),
            pess=ud_grade_pess,
            order_in=ud_grade_order_in,
            order_out=ud_grade_order_out,
            power=ud_grade_power,
        )

    return map_hi


@partial(jax.jit,
         static_argnames=("method", "nside", "lmax", "lcut",
                          "kernel_width_arcmin", "kernel_width_pixels",
                          "smoothing_interpretation", "iter"))
def deconvolve_map(
    hmap: Array,
    method: str,
    nside: int,
    *,
    lmax: Optional[int] = None,
    lcut: Optional[int] = None,
    kernel_width_arcmin: Optional[float] = None,
    kernel_width_pixels: Optional[float] = None,
    smoothing_interpretation: str = "fwhm",
    iter: int = 0,
    w_floor: float = 1e-8,
) -> Array:
    """Deconvolve the HEALPix mass-assignment window from a painted map.

    Painting particles onto a HEALPix map convolves the field with the
    assignment window ``W_l``. This removes one factor of that window at the
    map / ``a_lm`` level: ``map2alm`` → divide ``a_lm`` by ``W_l`` → ``alm2map``.

    Window per method (mirrors ``notebooks/09-Spherical_Deconvolution.ipynb``):

    - ``'ngp'``          : ``W_l = pixwin(nside)`` (the HEALPix pixel window).
    - ``'rbf_neighbor'`` : ``W_l = pixwin(nside) * B_l`` with the Gaussian beam
      ``B_l = exp(-l(l+1) sigma^2 / 2)``. ``sigma`` is resolved from the
      ``kernel_width_*`` arguments by the *same* helper the RBF painter uses, so
      the painting kernel and this beam cancel -- pass the **same** width you
      painted with.
    - ``'bilinear'``     : raises ``NotImplementedError``. Bilinear interpolation's
      effective window is position-dependent (not isotropic), so there is no
      closed-form per-``l`` ``B_l`` to divide out; deconvolve it empirically
      instead (measure ``W_l`` from a reference map -- see the notebook).

    Because ``1/W_l`` diverges as ``W_l → 0`` near the band limit (the spherical
    analogue of high-k ringing), modes with ``W_l <= w_floor`` are zeroed, and an
    optional ``lcut`` truncates the inverse window above a safe multipole.

    Parameters
    ----------
    hmap : ndarray
        Input HEALPix map in RING ordering, shape ``(12*nside**2,)``.
    method : {'ngp', 'rbf_neighbor', 'bilinear'} (case-insensitive)
        Painting scheme whose window to remove.
    nside : int
        HEALPix nside of ``hmap``.
    lmax : int, optional
        Maximum multipole. Default ``3*nside - 1`` (pixwin length). Must satisfy
        ``lmax >= 2*nside - 1`` for the s2fft transform.
    lcut : int, optional
        If given, zero the inverse window above ``lcut`` (extra high-l safety).
    kernel_width_arcmin, kernel_width_pixels, smoothing_interpretation :
        RBF beam width (used only for ``'rbf_neighbor'``); must match painting.
        ``kernel_width_pixels`` is in HEALPix pixels (a float; ``0.5`` = half a pixel).
    iter : int
        ``map2alm`` iterations. Default 0 (the only value the published
        ``jax_healpy`` supports; newer builds allow >0 for better accuracy).
    w_floor : float
        Modes with ``W_l <= w_floor`` are dropped (avoid 1/0 amplification).

    Returns
    -------
    ndarray
        The deconvolved HEALPix map at ``nside``.

    Notes
    -----
    Enable 64-bit (``jax.config.update('jax_enable_x64', True)``) for accurate
    transforms at high ``l``. The returned field is *sharpened* and may dip
    slightly negative near the band limit -- it is a window-corrected field, not
    a strictly non-negative density.

    Examples
    --------
    NGP and RBF have closed-form windows, so deconvolution is a single call::

        m_deconv = deconvolve_map(m_ngp, method="ngp", nside=nside, lmax=lmax)

    Bilinear has no analytic window, so estimate it **empirically** from a
    reference painted with a scheme whose window *is* known (NGP). Both maps see
    the same underlying field, and the window enters the power spectrum squared,
    so the bilinear window is ``W_l = pixwin * sqrt(C_l^bilinear / C_l^NGP)``::

        import healpy as hp, numpy as np, jax.numpy as jnp
        import jax_healpy as jhp
        from jaxpm.spherical import paint_particles_spherical

        # Paint the same particles two ways (same observer / shell / box).
        common = dict(nside=nside, observer_position=obs, R_min=R_min,
                      R_max=R_max, box_size=box_size, mesh_shape=mesh_shape)
        m_ngp  = paint_particles_spherical(pos, method="ngp",      **common)
        m_bili = paint_particles_spherical(pos, method="bilinear", **common)

        # Measure both auto-spectra on the overdensity.
        od = lambda m: np.asarray(m) / np.mean(np.asarray(m)) - 1.0
        cl_ngp  = hp.anafast(od(m_ngp),  lmax=lmax)
        cl_bili = hp.anafast(od(m_bili), lmax=lmax)

        # Empirical bilinear window: pixel window x sqrt(power ratio).
        pix = np.asarray(hp.pixwin(nside, lmax=lmax))
        with np.errstate(divide="ignore", invalid="ignore"):
            W = pix * np.sqrt(np.clip(cl_bili / cl_ngp, 0.0, None))

        # Deconvolve: divide a_lm by W_l, with the same high-l guard as above.
        inv = np.where(W > 1e-8, 1.0 / np.where(W > 1e-8, W, 1.0), 0.0)
        alm = jhp.map2alm(od(m_bili), lmax=lmax, iter=0) * jnp.asarray(inv)[:, None]
        m_bili_deconv = jnp.real(jhp.alm2map(alm, nside=nside, lmax=lmax))

    The empirical window is only trustworthy where ``cl_ngp`` is signal- (not
    shot-noise-) dominated; bin ``W_l`` in ``l`` and cap it with ``lcut`` at high
    ``l``.
    """
    import healpy as hp  # local: only deconvolution needs healpy's pixwin

    method_upper = method.strip().upper()
    nside = int(nside)
    if lmax is None:
        lmax = 3 * nside - 1
    lmax = int(lmax)

    # Host-side window W_l (static in nside/lmax/width): pixwin [* Gaussian beam].
    W = np.asarray(hp.pixwin(nside, lmax=lmax), dtype=np.float64)
    if method_upper == "NGP":
        pass
    elif method_upper == "RBF_NEIGHBOR":
        sigma = float(
            _smoothing_sigma_rad(
                nside,
                kernel_width_arcmin=kernel_width_arcmin,
                kernel_width_pixels=kernel_width_pixels,
                smoothing_interpretation=smoothing_interpretation,
            ))
        ell = np.arange(W.shape[0])
        W = W * np.exp(-ell * (ell + 1) * sigma**2 / 2.0)
    elif method_upper == "BILINEAR":
        raise NotImplementedError(
            "Bilinear painting has no closed-form deconvolution: its effective "
            "window is position-dependent (not isotropic), so there is no single "
            "B_l to divide out. Deconvolve it empirically instead: paint the same "
            "particles with NGP, measure both auto-spectra, and use "
            "W_l = pixwin * sqrt(cl_bilinear / cl_ngp) as the window (the sqrt is "
            "because the window enters the power spectrum squared). See the "
            "'Examples' section of deconvolve_map's docstring for runnable code."
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose 'ngp', 'rbf_neighbor', or "
            "'bilinear'.")

    # Inverse window with the high-l blow-up guard. Built with jnp so a *traced*
    # w_floor works under jit; ``W`` is a compile-time constant (static nside/lmax/
    # width) and ``lcut`` is static, so the slice update is also constant-folded.
    W = jnp.asarray(W)
    safe = W > w_floor
    inv = jnp.where(safe, 1.0 / jnp.where(safe, W, 1.0), 0.0)
    if lcut is not None:
        inv = inv.at[int(lcut) + 1:].set(0.0)

    # Harmonic-space deconvolution (jax_healpy transforms; almxfl applied inline).
    alm = jhp.map2alm(hmap, lmax=lmax, iter=int(iter))
    alm = alm * inv[:, None]  # s2fft ordering (L, 2L-1): broadcast W_l over m
    # jnp.real: some jax_healpy builds return a complex map with ~1e-16 imag noise.
    return jnp.real(jhp.alm2map(alm, nside=nside, lmax=lmax))


@partial(jax.jit, static_argnames=("nside", ))
def spherical_visibility_mask(
        nside: int,
        observer_position: Array,  # normalized coords in [0,1]^3
) -> Array:
    """
    Geometric visibility mask using only nside and observer_position.

    Assumptions
    -----------
    - Simulation volume is the unit cube [0,1]^3 (axis-aligned).
    - observer_position is given in normalized coordinates.
    - A pixel is 'visible' if a ray from the observer through that pixel's
      direction hits the cube for some t > 0.

    Returns
    -------
    mask : float32 ndarray of shape (12 * nside^2,)
        1.0 where visible, 0.0 otherwise.
    """
    obs = jnp.asarray(observer_position, dtype=jnp.float32)  # (3,)
    obs = jnp.clip(obs, 0.0, 1.0)
    bmin = jnp.zeros((3, ), dtype=jnp.float32)
    bmax = jnp.ones((3, ), dtype=jnp.float32)

    npix = jhp.nside2npix(nside)
    ipix = jnp.arange(npix, dtype=jnp.int64)
    # Pixel center directions (unit vectors), shape (npix, 3)
    dirs = jhp.pix2vec(nside, ipix)

    # Robust slab intersection (vectorized).
    # For axes where dir == 0, treat as parallel: if obs is inside the slab,
    # set t to (-inf, +inf); else to (+inf, -inf) so it won't intersect.
    eps = jnp.float32(1e-12)
    inside_axis = (obs >= bmin) & (obs <= bmax)  # (3,)
    dir_nz = jnp.abs(dirs) > eps  # (npix, 3)

    t1 = jnp.where(dir_nz, (bmin - obs) / dirs,
                   jnp.where(inside_axis, -jnp.inf, jnp.inf))
    t2 = jnp.where(dir_nz, (bmax - obs) / dirs,
                   jnp.where(inside_axis, jnp.inf, -jnp.inf))

    tmin = jnp.max(jnp.minimum(t1, t2), axis=-1)  # (npix,)
    tmax = jnp.min(jnp.maximum(t1, t2), axis=-1)  # (npix,)

    # Visible if the forward ray intersects: tmax >= max(tmin, 0)
    visible = tmax > jnp.maximum(tmin, jnp.float32(0.0))
    return visible.astype(jnp.float32)
