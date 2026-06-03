from functools import partial
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax_healpy as jhp

Array = jnp.ndarray


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
    kernel_width_arcmin : float
        Width of the Gaussian smoothing kernel in arcminutes.
        Larger values → more smoothing (blurrier maps).
        Smaller values → less smoothing (sharper maps).
    smoothing_interpretation : {"fwhm", "sigma", "2sigma"}
        Interpretation of kernel_width_arcmin:
        - 'fwhm': kernel_width_arcmin is the full-width at half-maximum
        - 'sigma': kernel_width_arcmin is the standard deviation
        - '2sigma': kernel_width_arcmin is 2× the standard deviation
    sharding : jax.sharding.Sharding, optional
        Sharding information for distributed computation. If provided, the HEALPix map
        will be allocated with the specified sharding

    Returns
    -------
    healpix_map : ndarray
        HEALPix density map
    """
    if kernel_width_arcmin is not None:
        smoothing_rad = jnp.asarray(kernel_width_arcmin) * (jnp.pi /
                                                            180.0) / 60.0

        if smoothing_interpretation == "fwhm":
            sigma = smoothing_rad / 2.355
        elif smoothing_interpretation == "2sigma":
            sigma = smoothing_rad / 2.0
        elif smoothing_interpretation == "sigma":
            sigma = smoothing_rad
        else:
            raise ValueError(
                "smoothing_interpretation must be one of 'fwhm', 'sigma', or '2sigma'"
            )
    else:
        pixel_scale = jnp.asarray(jhp.nside2resol(nside))
        sigma = pixel_scale / 2.0

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
    kernel_width_arcmin : float
        Width of the Gaussian kernel in arcminutes for the RBF method.
        Larger values → more smoothing. Smaller values → less smoothing.
    smoothing_interpretation : {"fwhm", "sigma", "2sigma"}
        Interpretation of kernel_width_arcmin for the RBF method:
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
