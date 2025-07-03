from functools import partial

import jax
import jax.numpy as jnp
import jax_healpy as jhp

from jaxpm.distributed import uniform_particles


@partial(jax.jit, static_argnames=('nside', 'mesh_shape', 'box_size'))
def paint_particles_spherical(positions,
                              nside,
                              observer_position,
                              R_min,
                              R_max,
                              box_size,
                              mesh_shape,
                              weights=None):
    """
    Directly bin particles onto HEALPix spherical maps without intermediate
    3D Cartesian mesh. This avoids double binning artifacts.

    Parameters
    ----------
    positions : ndarray, shape (..., 3)
        Particle positions in simulation coordinates
    nside : int
        HEALPix nside parameter
    observer_position : ndarray, shape (3,)
        Observer position in comoving  coordinates
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
    # Check resolution warning
    res_deg = jhp.nside2resol(nside, arcmin=True) / 60
    box_diagonal = jnp.sqrt(sum([bs**2 for bs in box_size]))
    typical_angular_scale = jnp.degrees(jnp.arctan(min(box_size) / box_diagonal))
    
    #if res_deg > typical_angular_scale:
    #    jax.debug.print(
    #        "WARNING: HEALPix resolution ({res_deg} deg) is larger than typical box angular scale ({typical_angular_scale:} deg). "
    #        "Consider decreasing nside from {nside} or increasing box size. "
    #        "Current box size: {box_size}, nside resolution: {res_deg*60:.2f} arcmin" , res_deg=res_deg,
    #        typical_angular_scale=typical_angular_scale,
    #        nside=nside, box_size=box_size
    #    )
    
    if weights is None:
        weights = jnp.ones(positions.shape[:-1])

    # Convert particle positions from simulation coordinates to physical coordinates
    # by scaling with box_size and mesh_shape
    positions = positions * jnp.array(box_size) / jnp.array(mesh_shape)

    # Compute relative positions from observer
    rel_positions = positions - jnp.asarray(observer_position)

    # Convert to spherical coordinates
    x, y, z = rel_positions[..., 0], rel_positions[..., 1], rel_positions[...,
                                                                          2]

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
    
    # Calculate volume per pixel in spherical shell
    pixel_solid_angle = 4 * jnp.pi / npix  # steradians per pixel
    R_center = 0.5 * (R_min + R_max)
    shell_thickness = R_max - R_min
    shell_volume_per_pixel = pixel_solid_angle * R_center**2 * shell_thickness

    # Convert particle counts to density (particles per unit volume)
    healpix_map = healpix_map / shell_volume_per_pixel

    return healpix_map


@partial(jax.jit,
         static_argnames=('nside', 'fov', 'center_radec', 'd_R', 'box_size'))
def paint_spherical(volume, nside, fov, center_radec, observer_position,
                    box_size, R, d_R):
    width, height, depth = volume.shape
    ra0, dec0 = center_radec
    fov_width, fov_height = fov

    pixel_scale_x = fov_width / width
    pixel_scale_y = fov_height / height

    res_deg = jhp.nside2resol(nside, arcmin=True) / 60
    if pixel_scale_x > res_deg or pixel_scale_y > res_deg:
        print(
            f"WARNING Pixel scale ({pixel_scale_x:.4f} deg, {pixel_scale_y:.4f} deg) is larger than the Healpy resolution ({res_deg:.4f} deg). Increase the field of view or decrease the nside."
        )

    y_idx, x_idx = jnp.indices((height, width))
    ra_grid = ra0 + x_idx * pixel_scale_x
    dec_grid = dec0 + y_idx * pixel_scale_y

    ra_flat = ra_grid.flatten() * jnp.pi / 180.0
    dec_flat = dec_grid.flatten() * jnp.pi / 180.0
    R_s = jnp.arange(0, d_R, 1.0) + R

    XYZ = R_s.reshape(-1, 1, 1) * jhp.ang2vec(ra_flat, dec_flat, lonlat=False)
    observer_position = jnp.array(observer_position)
    # Convert observer position from box units to grid units
    observer_position = observer_position / jnp.array(box_size) * jnp.array(
        volume.shape)

    coords = XYZ + jnp.asarray(observer_position)[jnp.newaxis, jnp.newaxis, :]

    pixels = jhp.ang2pix(nside, ra_flat, dec_flat, lonlat=False)

    npix = jhp.nside2npix(nside)

    @partial(jax.vmap, in_axes=(0, None, None))
    def interpolate_volume(coords, volume, pixels):
        voxels = jax.scipy.ndimage.map_coordinates(volume, coords.T, order=1)
        sums = jnp.bincount(pixels, weights=voxels, length=npix)
        return sums

    sum_map = interpolate_volume(coords, volume, pixels).sum(axis=0)
    counts = jnp.bincount(pixels, length=npix)
    sum_map = jnp.where(counts > 0, sum_map / counts, jhp.UNSEEN)

    return sum_map
