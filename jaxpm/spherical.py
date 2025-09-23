import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np


def paint_particles_spherical(positions,
                              nside,
                              observer_position,
                              R_min,
                              R_max,
                              box_size,
                              mesh_shape,
                              weights=None):
    """
    Paint particles onto HEALPix spherical maps using Nearest Grid Point (NGP) scheme.
    
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
    box_size : float
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

    # Convert particle positions from simulation coordinates to physical coordinates
    # by scaling with box_size and mesh_shape
    positions = positions * jnp.array(box_size) / jnp.array(mesh_shape)

    # Compute relative positions from observer
    rel_positions = positions - jnp.asarray(observer_position)

    # Convert to spherical coordinates
    x, y, z = rel_positions[..., 0], rel_positions[..., 1], rel_positions[..., 2]

    # Comoving distance from observer
    r = jnp.sqrt(x**2 + y**2 + z**2)

    # Apply distance cuts
    distance_mask = (r >= R_min) & (r <= R_max)

    # Compute angular coordinates (theta, phi in spherical coordinates)
    # theta = polar angle from z-axis, phi = azimuthal angle
    theta = jnp.arccos(jnp.clip(z / (r + 1e-10), -1, 1))
    phi = jnp.arctan2(y, x)

    # Convert to HEALPix pixel indices
    pixels = jhp.ang2pix(nside, theta.flatten(), phi.flatten())

    # Apply distance mask to weights
    masked_weights = (weights * distance_mask).flatten()

    # Bin particles into HEALPix pixels
    npix = jhp.nside2npix(nside)
    healpix_map = jnp.bincount(pixels, weights=masked_weights, length=npix)

    # Calculate the volume of the spherical shell
    shell_volume = 4/3 * jnp.pi * (R_max**3 - R_min**3)

    # Calculate the volume per pixel
    pixel_volume = shell_volume / npix

    # Convert particle counts to physical density
    density_map = healpix_map / pixel_volume

    return density_map
