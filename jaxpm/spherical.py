from functools import partial
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax_healpy as jhp

Array = jnp.ndarray


@partial(jax.jit, static_argnames=("nside", ))
def paint_particles_spherical_ngp(
    positions: Array,
    nside: int,
    observer_position: Union[Array, jnp.ndarray],
    R_min: float,
    R_max: float,
    box_size: Union[float, Array, jnp.ndarray],
    mesh_shape: Tuple[int, int, int],
    weights: Optional[Array] = None,
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

    Returns
    -------
    healpix_map : ndarray
        HEALPix density map
    """
    if weights is None:
        weights = jnp.ones(positions.shape[:-1])

    # Convert particle positions to physical coordinates
    positions_phys = positions * jnp.array(box_size) / jnp.array(mesh_shape)

    # Compute relative positions from observer
    rel_positions = positions_phys - jnp.asarray(observer_position)

    # Comoving distance from observer
    r = jnp.sqrt(jnp.sum(rel_positions**2, axis=-1))

    # Apply distance cuts - use masking instead of boolean indexing
    distance_mask = (r >= R_min) & (r <= R_max)

    # Flatten arrays for processing (keep static shapes)
    rel_positions_flat = rel_positions.reshape(-1, 3)
    r_flat = r.flatten()
    weights_flat = weights.flatten()
    distance_mask_flat = distance_mask.flatten()

    # Apply mask to weights (static shape preserved)
    masked_weights = jnp.where(distance_mask_flat, weights_flat, 0.0)

    # Safe division to avoid division by zero
    r_safe = jnp.where(r_flat > 1e-10, r_flat, 1e-10)
    unit_vecs = rel_positions_flat / r_safe[..., None]

    # Convert unit vectors to angles using jax_healpy
    theta, phi = jhp.vec2ang(unit_vecs)

    # Convert to HEALPix pixel indices
    pixels = jhp.ang2pix(nside, theta, phi)

    # Bin particles into HEALPix pixels
    npix = jhp.nside2npix(nside)
    healpix_map = jnp.bincount(pixels, weights=masked_weights, length=npix)

    # Calculate volume per pixel in spherical shell (exact shell volume)
    pixel_solid_angle = 4 * jnp.pi / npix  # steradians per pixel
    shell_volume_per_pixel = pixel_solid_angle * (R_max**3 - R_min**3) / 3.0

    # Convert particle counts to density (particles per unit volume)
    return healpix_map / shell_volume_per_pixel


@partial(jax.jit, static_argnames=("nside", ))
def paint_particles_spherical_bilinear(
    positions: Array,
    nside: int,
    observer_position: Union[Array, jnp.ndarray],
    R_min: float,
    R_max: float,
    box_size: Union[float, Array, jnp.ndarray],
    mesh_shape: Tuple[int, int, int],
    weights: Optional[Array] = None,
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

    Returns
    -------
    healpix_map : ndarray
        HEALPix density map
    """
    if weights is None:
        weights = jnp.ones(positions.shape[:-1])

    # Convert particle positions to physical coordinates
    positions_phys = positions * jnp.array(box_size) / jnp.array(mesh_shape)

    # Compute relative positions from observer
    rel_positions = positions_phys - jnp.asarray(observer_position)

    # Comoving distance from observer
    r = jnp.sqrt(jnp.sum(rel_positions**2, axis=-1))

    # Apply distance cuts using masking (no boolean indexing)
    distance_mask = (r >= R_min) & (r <= R_max)

    # Flatten arrays (keep static shapes)
    rel_positions_flat = rel_positions.reshape(-1, 3)
    r_flat = r.flatten()
    weights_flat = weights.flatten()
    distance_mask_flat = distance_mask.flatten()

    # Apply mask to weights (static shape preserved)
    masked_weights = jnp.where(distance_mask_flat, weights_flat, 0.0)

    # Safe division to avoid division by zero
    r_safe = jnp.where(r_flat > 1e-10, r_flat, 1e-10)
    unit_vecs = rel_positions_flat / r_safe[..., None]

    # Convert unit vectors to spherical coordinates
    theta, phi = jhp.vec2ang(unit_vecs)

    # Get bilinear interpolation weights and pixel indices
    pixels, interp_weights = jhp.get_interp_weights(nside, theta, phi)
    # check if normalized
    weight_sums = jnp.sum(interp_weights, axis=0)
    norm_bool = jnp.allclose(weight_sums, 1.0)

    # Initialize HEALPix map
    npix = jhp.nside2npix(nside)
    healpix_map = jnp.zeros(npix)

    contributions = interp_weights * masked_weights[None, :]
    # Calculate contributions for each of the 4 nearest pixels
    healpix_map = healpix_map.at[pixels].add(contributions)

    # Apply shell-volume normalization
    pixel_area = 4 * jnp.pi / npix
    shell_vol_per_pix = pixel_area * (R_max**3 - R_min**3) / 3
    return healpix_map / shell_vol_per_pix


@partial(jax.jit, static_argnames=("nside", ))
def paint_particles_spherical_rbf_neighbor(
    positions: Array,
    nside: int,
    observer_position: Union[Array, jnp.ndarray],
    R_min: float,
    R_max: float,
    box_size: Union[float, Array, jnp.ndarray],
    mesh_shape: Tuple[int, int, int],
    weights: Optional[Array] = None,
    sigma_fixed: float = 0.05,
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
        Observer position in comoving coordinates
    R_min, R_max : float
        Minimum and maximum comoving distance range to include
    box_size : float or array
        Size of the simulation box in physical units
    mesh_shape : tuple
        Shape of the simulation mesh (nx, ny, nz)
    weights : ndarray, optional
        Particle weights (default: uniform weights)
    sigma_fixed : float
        Fixed smoothing width parameter

    Returns
    -------
    healpix_map : ndarray
        HEALPix density map
    """
    if weights is None:
        weights = jnp.ones(positions.shape[:-1])

    # Convert particle positions to physical coordinates
    positions_phys = positions * jnp.array(box_size) / jnp.array(mesh_shape)

    # Compute relative positions from observer
    rel_positions = positions_phys - jnp.asarray(observer_position)

    # Comoving distance from observer
    r = jnp.sqrt(jnp.sum(rel_positions**2, axis=-1))

    # Apply distance cuts using masking (no boolean indexing)
    distance_mask = (r >= R_min) & (r <= R_max)

    # Flatten arrays (keep static shapes)
    rel_positions_flat = rel_positions.reshape(-1, 3)
    r_flat = r.flatten()
    weights_flat = weights.flatten()
    distance_mask_flat = distance_mask.flatten()

    # Apply mask to weights (static shape preserved)
    masked_weights = jnp.where(distance_mask_flat, weights_flat, 0.0)

    # Safe division to avoid division by zero
    r_safe = jnp.where(r_flat > 1e-10, r_flat, 1e-10)
    unit_vecs = rel_positions_flat / r_safe[..., None]

    # Convert unit vectors to spherical coordinates
    theta, phi = jhp.vec2ang(unit_vecs)

    # Get 9-pixel stencils: center + 8 neighbors
    pix9 = jhp.get_all_neighbours(nside,
                                  theta,
                                  phi,
                                  nest=False,
                                  get_center=True)

    # Get unit vectors for all 9 pixels
    flat_pix = pix9.reshape(-1)
    vecs = jhp.pix2vec(nside, flat_pix).reshape(9, -1, 3)
    # NaNs can arise for neighbors with index -1 (non-existent); keep vectors but mask later.
    vecs = jnp.nan_to_num(vecs)
    # Compute angular separations for all (neighbor, particle) pairs
    dots = jnp.einsum("ij,kij->ki", unit_vecs, vecs)
    gamma = jnp.arccos(jnp.clip(dots, -1.0, 1.0))

    # Gaussian kernel weights
    kernel_weights = jnp.exp(
        -(gamma**2) / (2 * sigma_fixed**2)) / (2 * jnp.pi * sigma_fixed**2)
    # Check if kernel weights are normalized per particle
    weight_sums = jnp.sum(kernel_weights, axis=0)
    norm_bool = jnp.allclose(weight_sums, 1.0)

    # Mask invalid neighbors (pix == -1) and renormalize per particle to conserve mass
    valid_mask = (pix9 != -1)
    kernel_weights_masked = jnp.where(valid_mask, kernel_weights, 0.0)
    weight_sum = jnp.sum(kernel_weights_masked, axis=0)
    # Safe normalization: if no valid neighbors, keep zeros
    norm_kernel = jnp.where(weight_sum[None, :] > 0.0,
                            kernel_weights_masked / weight_sum[None, :], 0.0)
    # Check normalization again
    norm_sums = jnp.sum(norm_kernel, axis=0)
    norm_bool2 = jnp.allclose(norm_sums, 1.0)

    # Initialize HEALPix map size
    npix = jhp.nside2npix(nside)

    # Weight by particle weight; kernel sums to 1 per particle
    pixel_area = 4.0 * jnp.pi / npix
    contrib = norm_kernel * masked_weights[None, :]

    # Scatter contributions into a map
    idx = pix9.reshape(-1)
    val = contrib.reshape(-1)
    # Avoid negative indices by zeroing their contributions and redirecting index to 0
    valid_flat = (idx != -1)
    idx_safe = jnp.where(valid_flat, idx, 0)
    val_safe = jnp.where(valid_flat, val, 0.0)
    healpix_map = jnp.zeros(npix).at[idx_safe].add(val_safe)

    # Apply shell-volume normalization
    shell_vol_per_pix = pixel_area * (R_max**3 - R_min**3) / 3
    return healpix_map / shell_vol_per_pix


@partial(jax.jit,
         static_argnames=("nside", "method", "udgrade_order_in",
                          "udgrade_order_out", "udgrade_power", "udgrade_pess",
                          "paint_nside"))
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
    sigma_fixed: float = 0.05,
    # High-resolution painting option
    paint_nside: Optional[int] = None,
    udgrade_power: float = 0.0,
    udgrade_order_in: str = "RING",
    udgrade_order_out: str = "RING",
    udgrade_pess: bool = False,
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
    method : str
        Painting method: 'ngp', 'bilinear', or 'rbf_neighbor'
    sigma_fixed : float
        Fixed smoothing width parameter for RBF method
    paint_nside : int, optional
        Internal resolution to paint at. If None, equals nside.
    udgrade_power : float
        Power parameter for udgrade
    udgrade_order_in : str
        Input pixel ordering for udgrade
    udgrade_order_out : str
        Output pixel ordering for udgrade
    udgrade_pess : bool
        Pessimistic flag for udgrade

    Returns
    -------
    healpix_map : ndarray
        HEALPix density map at resolution nside
    """
    method_upper = method.strip().upper()

    # Determine internal nside for painting
    internal_nside = int(paint_nside) if paint_nside is not None else int(
        nside)

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
            sigma_fixed=sigma_fixed,
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose 'ngp', 'bilinear', or 'rbf_neighbor'."
        )

    # If internal resolution differs from target, up/down-grade
    if internal_nside != int(nside):
        return jhp.udgrade(
            map_hi,
            int(nside),
            pess=udgrade_pess,
            order_in=udgrade_order_in,
            order_out=udgrade_order_out,
            power=udgrade_power,
        )

    return map_hi
