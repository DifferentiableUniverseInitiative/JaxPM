from functools import partial

import healpy as hp
import jax
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np


def gaussian_kernel_angle(gamma, sigma):
    """Gaussian kernel for angular separations.

    Parameters
    ----------
    gamma : float or array
        Angular separation in radians
    sigma : float
        Kernel width parameter

    Returns
    -------
    kernel_value : float or array
        Gaussian kernel value
    """
    return np.exp(-(gamma**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)


def sigma_from_h(h, R_center):
    """Convert smoothing length h to angular sigma.

    Parameters
    ----------
    h : float
        Smoothing length
    R_center : float
        Center distance of the shell

    Returns
    -------
    sigma : float
        Angular sigma parameter
    """
    return h / R_center


def pixels_within_radius(nside, vec, gamma_cut):
    """Find HEALPix pixels within angular radius.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter
    vec : array, shape (3,)
        Unit 3-vector of the particle
    gamma_cut : float
        Angular radius cutoff in radians

    Returns
    -------
    pixel_indices : array
        HEALPix pixel indices within the radius
    """
    return hp.query_disc(nside, vec, gamma_cut, inclusive=True, nest=False)


def angsep_vecs(vec1, vec2):
    """Compute angular separation between unit vectors.

    Parameters
    ----------
    vec1, vec2 : array, shape (3,)
        Unit 3-vectors

    Returns
    -------
    gamma : float
        Angular separation in radians
    """
    return np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))


def _healpix_cic_weights(nside, theta, phi):
    """
    Compute CIC (Cloud-in-Cell) weights for HEALPix pixels.

    This implements a simplified differentiable CIC scheme for spherical coordinates.
    Instead of using exact HEALPix face projections, we use a smooth approximation
    based on angular distances to nearby pixels.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter
    theta : array_like
        Colatitude in radians (0 to pi)
    phi : array_like
        Longitude in radians (0 to 2pi)

    Returns
    -------
    pixel_indices : array, shape (4, N)
        Four neighbor pixel indices for each particle
    weights : array, shape (4, N)
        CIC weights for each neighbor pixel
    """
    # Convert to HEALPix pixel coordinates
    host_pixels = jhp.ang2pix(nside, theta, phi)

    # Get pixel centers for the host pixel
    theta_host, phi_host = jhp.pix2ang(nside, host_pixels)

    # Estimate pixel size in radians
    pixel_size = jnp.sqrt(4 * jnp.pi / (12 * nside**2))

    # Compute offsets from pixel center
    dtheta = theta - theta_host
    dphi = phi - phi_host

    # Handle phi wraparound
    dphi = jnp.where(dphi > jnp.pi, dphi - 2 * jnp.pi, dphi)
    dphi = jnp.where(dphi < -jnp.pi, dphi + 2 * jnp.pi, dphi)

    # Normalize to [0, 1] fractional coordinates within pixel
    # Use a smooth approximation for differentiability
    delta_u = jnp.clip(0.5 + dtheta / pixel_size, 0.0, 1.0 - 1e-7)
    delta_v = jnp.clip(0.5 + dphi / pixel_size, 0.0, 1.0 - 1e-7)

    # Create 4 neighbor pixels by using small angular offsets
    # This is a simplified approach - we create neighbors by shifting theta/phi
    offset = pixel_size * 0.25  # Quarter pixel offset

    # Define the 4 corners relative to host pixel
    theta_neighbors = jnp.array([
        theta_host - offset,  # SW
        theta_host - offset,  # SE
        theta_host + offset,  # NW
        theta_host + offset  # NE
    ])

    phi_neighbors = jnp.array([
        phi_host - offset,  # SW
        phi_host + offset,  # SE
        phi_host - offset,  # NW
        phi_host + offset  # NE
    ])

    # Convert neighbor coordinates to pixel indices
    # Clamp theta to valid range
    theta_neighbors = jnp.clip(theta_neighbors, 1e-8, jnp.pi - 1e-8)

    # Get pixel indices for each neighbor (broadcasting over particles)
    pixel_indices = jnp.array([
        jhp.ang2pix(nside, theta_neighbors[i], phi_neighbors[i])
        for i in range(4)
    ])

    # Compute CIC weights for the 4 corners
    # Order: SW, SE, NW, NE
    weights = jnp.array([
        (1 - delta_u) * (1 - delta_v),  # SW
        (1 - delta_u) * delta_v,  # SE
        delta_u * (1 - delta_v),  # NW
        delta_u * delta_v  # NE
    ])

    return pixel_indices, weights


def paint_particles_spherical_cic(positions,
                                  nside,
                                  observer_position,
                                  R_min,
                                  R_max,
                                  box_size,
                                  mesh_shape,
                                  weights=None):
    """
    Paint particles onto HEALPix spherical maps using differentiable CIC scheme.

    This implements the Cloud-in-Cell (CIC) mass-assignment scheme for spherical
    coordinates, making it differentiable unlike the NGP scheme.

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

    # Convert particle positions to physical coordinates
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

    # Compute angular coordinates
    theta = jnp.arccos(jnp.clip(z / (r + 1e-10), -1, 1))
    phi = jnp.arctan2(y, x)

    # Apply distance mask to weights
    masked_weights = (weights * distance_mask).flatten()

    # Get CIC weights for each particle
    pixel_indices, cic_weights = _healpix_cic_weights(nside, theta.flatten(),
                                                      phi.flatten())

    # Initialize HEALPix map
    npix = jhp.nside2npix(nside)
    healpix_map = jnp.zeros(npix)

    # Apply CIC weights to each of the 4 neighbors
    for i in range(4):
        # Get pixel indices and weights for this neighbor
        pix_idx = pixel_indices[i]  # Shape: (N,)
        cic_weight = cic_weights[i]  # Shape: (N,)

        # Combine distance mask and CIC weights
        combined_weights = masked_weights * cic_weight

        # Add contributions using scatter_add for differentiability
        healpix_map = healpix_map.at[pix_idx].add(combined_weights)

    # Calculate volume per pixel in spherical shell
    pixel_solid_angle = 4 * jnp.pi / npix  # steradians per pixel
    R_center = 0.5 * (R_min + R_max)
    shell_thickness = R_max - R_min
    shell_volume_per_pixel = pixel_solid_angle * R_center**2 * shell_thickness

    # Convert particle counts to density (particles per unit volume)
    healpix_map = healpix_map / shell_volume_per_pixel

    return healpix_map


def paint_particles_spherical_ngp(positions,
                                  nside,
                                  observer_position,
                                  R_min,
                                  R_max,
                                  box_size,
                                  mesh_shape,
                                  weights=None):
    """
    Paint particles onto HEALPix spherical maps using Nearest Grid Point (NGP) scheme.

    This is a non-differentiable method that assigns particles to the nearest pixel.

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
    # Original NGP implementation
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


def paint_particles_spherical_rbf(positions,
                                  nside,
                                  observer_position,
                                  R_min,
                                  R_max,
                                  box_size,
                                  mesh_shape,
                                  weights=None,
                                  sigma_fixed=None,
                                  gamma_cut_factor=3.0):
    """
    Paint particles onto HEALPix spherical maps using Radial Basis Function (RBF) scheme.

    This implements a smooth RBF kernel for particle painting, using numpy/healpy
    for non-differentiable but potentially more accurate calculations.

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
    sigma_fixed : float, optional
        Fixed smoothing width parameter. If None, uses adaptive smoothing.
    gamma_cut_factor : float, optional
        Cutoff factor for kernel support (default: 3.0)

    Returns
    -------
    healpix_map : ndarray
        HEALPix density map
    """
    if weights is None:
        weights = np.ones(positions.shape[:-1])

    # Convert from JAX arrays to numpy if needed
    positions = np.asarray(positions)
    weights = np.asarray(weights)
    observer_position = np.asarray(observer_position)

    # Convert particle positions to physical coordinates
    positions = positions * np.array(box_size) / np.array(mesh_shape)

    # Compute relative positions from observer
    rel_positions = positions - observer_position

    # Convert to spherical coordinates
    x, y, z = rel_positions[..., 0], rel_positions[..., 1], rel_positions[...,
                                                                          2]

    # Comoving distance from observer
    r = np.sqrt(x**2 + y**2 + z**2)

    # Apply distance cuts
    distance_mask = (r >= R_min) & (r <= R_max)

    # Initialize HEALPix map
    npix = hp.nside2npix(nside)
    healpix_map = np.zeros(npix)

    # Flatten arrays for easier processing
    positions_flat = positions.reshape(-1, 3)
    rel_positions_flat = rel_positions.reshape(-1, 3)
    r_flat = r.flatten()
    weights_flat = weights.flatten()
    distance_mask_flat = distance_mask.flatten()

    # Calculate shell center distance for sigma calculation
    R_center = 0.5 * (R_min + R_max)

    # Process each particle
    for i in range(len(positions_flat)):
        if not distance_mask_flat[i]:
            continue

        # Get particle properties
        r_p = r_flat[i]
        m_p = weights_flat[i]

        # Convert to unit vector
        vec_p = rel_positions_flat[i] / r_p

        # Determine smoothing width
        if sigma_fixed is not None:
            sigma_p = sigma_fixed
        else:
            # For now, use fixed sigma - adaptive can be added later
            sigma_p = sigma_fixed if sigma_fixed is not None else 0.1  # Default value

        # Cutoff radius for kernel support
        gamma_cut = gamma_cut_factor * sigma_p

        # Find pixels within cutoff radius
        pix_indices = pixels_within_radius(nside, vec_p, gamma_cut)

        # Calculate pixel area (constant for all pixels)
        pixel_area = 4 * np.pi / npix

        # Process each pixel within the cutoff
        for pix in pix_indices:
            # Get pixel center direction
            vec_i = np.array(hp.pix2vec(nside, pix))

            # Calculate angular separation
            gamma = angsep_vecs(vec_p, vec_i)

            # Calculate kernel weight
            w = gaussian_kernel_angle(gamma, sigma_p) * pixel_area

            # Add contribution to the map
            healpix_map[pix] += m_p * w

    # Apply shell-volume normalization
    pixel_area = 4 * np.pi / npix
    shell_vol_per_pix = pixel_area * (R_max**3 - R_min**3) / 3
    healpix_map /= shell_vol_per_pix

    return healpix_map


def paint_particles_spherical(positions,
                              nside,
                              observer_position,
                              R_min,
                              R_max,
                              box_size,
                              mesh_shape,
                              weights=None,
                              method='cic',
                              sigma_fixed=None,
                              gamma_cut_factor=3.0):
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
    method : str, optional
        Painting method: 'cic' for Cloud-in-Cell (differentiable), 'ngp' for
        Nearest Grid Point (non-differentiable), or 'rbf' for Radial Basis Function.
        Default is 'cic'.
    sigma_fixed : float, optional
        Fixed smoothing width parameter for RBF method. Only used when method='rbf'.
    gamma_cut_factor : float, optional
        Cutoff factor for RBF kernel support (default: 3.0). Only used when method='rbf'.

    Returns
    -------
    healpix_map : ndarray
        HEALPix density map
    """
    # Check particle density warning
    total_particles = jnp.prod(jnp.array(
        positions.shape[:-1]))  # Total number of particles
    npix = jhp.nside2npix(nside)  # Total HEALPix pixels

    # Estimate fraction of particles in the shell
    # Approximate shell volume vs box volume
    box_diagonal = jnp.sqrt(jnp.sum(jnp.array(box_size)**2))
    max_distance = box_diagonal / 2  # Maximum possible distance from center
    shell_volume_fraction = ((R_max**3 - R_min**3) / 3) / (max_distance**3 / 3)
    shell_volume_fraction = jnp.minimum(shell_volume_fraction,
                                        1.0)  # Cap at 1.0

    estimated_particles_in_shell = total_particles * shell_volume_fraction
    particles_per_pixel = estimated_particles_in_shell / npix

    # Warn if particles per pixel is too low (threshold: 1 particle per pixel)
    min_particles_per_pixel = 1.0
    jax.lax.cond(
        particles_per_pixel < min_particles_per_pixel, lambda: jax.debug.print(
            "WARNING: Low particle density detected! "
            "Estimated {particles_per_pixel} particles per pixel (threshold: {min_threshold}). "
            "This may result in shot noise and low statistical power. "
            "Consider: increasing mesh resolution, decreasing nside from {nside}, or using a thicker shell. "
            "Total particles: {total_particles}, Shell fraction: {shell_fraction}, Pixels: {npix}",
            particles_per_pixel=particles_per_pixel,
            min_threshold=min_particles_per_pixel,
            nside=nside,
            total_particles=total_particles,
            shell_fraction=shell_volume_fraction,
            npix=npix), lambda: None)

    # Choose method
    if method.upper() == 'CIC':
        # Apply JIT compilation for CIC method
        jit_func = jax.jit(paint_particles_spherical_cic,
                           static_argnames=('nside', 'mesh_shape', 'box_size'))
        return jit_func(positions, nside, observer_position, R_min, R_max,
                        box_size, mesh_shape, weights)
    elif method.upper() == 'NGP':
        # Apply JIT compilation for NGP method
        jit_func = jax.jit(paint_particles_spherical_ngp,
                           static_argnames=('nside', 'mesh_shape', 'box_size'))
        return jit_func(positions, nside, observer_position, R_min, R_max,
                        box_size, mesh_shape, weights)
    elif method.upper() == 'RBF':
        # RBF method uses numpy/healpy - no JIT compilation
        return paint_particles_spherical_rbf(positions, nside,
                                             observer_position, R_min, R_max,
                                             box_size, mesh_shape, weights,
                                             sigma_fixed, gamma_cut_factor)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose 'cic', 'ngp', or 'rbf'.")
