import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jax_cosmo.constants as constants
from jax.scipy.ndimage import map_coordinates

from jaxpm.distributed import uniform_particles
from jaxpm.painting import cic_paint, cic_paint_2d, cic_paint_dx
from jaxpm.spherical import paint_particles_spherical
from jaxpm.utils import gaussian_smoothing


def density_plane_fn(box_shape,
                     box_size,
                     density_plane_width,
                     density_plane_npix,
                     sharding=None):

    def f(t, y, args):
        positions = y
        cosmo = args
        nx, ny, nz = box_shape

        # Converts time t to comoving distance in voxel coordinates
        w = density_plane_width / box_size[2] * box_shape[2]
        center = jc.background.radial_comoving_distance(
            cosmo, t) / box_size[2] * box_shape[2]
        # Clear workspace to avoid memory issues and tracer leaks
        # due to the caching system in jax-cosmo
        cosmo._workspace = {}
        positions = uniform_particles(box_shape) + positions
        xy = positions[..., :2]
        d = positions[..., 2]

        # Apply 2d periodic conditions
        xy = jnp.mod(xy, nx)

        # Rescaling positions to target grid
        xy = xy / nx * density_plane_npix
        # Selecting only particles that fall inside the volume of interest
        weight = jnp.where((d > (center - w / 2)) & (d <= (center + w / 2)),
                           1.0, 0.0)
        # Painting density plane
        zero_mesh = jnp.zeros([density_plane_npix, density_plane_npix])
        # Apply sharding in order to recover sharding when taking gradients
        if sharding is not None:
            xy = jax.lax.with_sharding_constraint(xy, sharding)
        # Apply CIC painting
        density_plane = cic_paint_2d(zero_mesh, xy, weight)

        # Calculate physical volume per pixel
        pixel_area = (box_size[0] / density_plane_npix) * (box_size[1] /
                                                           density_plane_npix)
        shell_thickness_physical = density_plane_width  # Already in physical units
        pixel_volume = pixel_area * shell_thickness_physical

        # Convert counts to density (particles per unit volume)
        density_plane = density_plane / pixel_volume

        return density_plane

    return f


def spherical_density_fn(mesh_shape,
                         box_size,
                         nside,
                         observer_position,
                         density_plane_width,
                         method="RBF_NEIGHBOR",
                         kernel_width_arcmin=None,
                         sharding=None):

    def f(t, y, args):
        positions = y
        cosmo = args

        positions = uniform_particles(mesh_shape) + positions

        # Calculate comoving distance range for this shell
        r_center = jc.background.radial_comoving_distance(cosmo, t)
        # Clear workspace to avoid memory issues
        # due to the caching system in jax-cosmo
        cosmo._workspace = {}
        r_max = jnp.clip(r_center + density_plane_width / 2, 0, box_size[2])
        r_min = jnp.clip(r_center - density_plane_width / 2, 0, box_size[2])

        if sharding is not None:
            positions = jax.lax.with_sharding_constraint(positions, sharding)

        # Paint particles in this shell onto a HEALPix map
        spherical_map = paint_particles_spherical(
            positions,
            nside=nside,
            method=method,
            kernel_width_arcmin=kernel_width_arcmin,
            observer_position=observer_position,
            R_min=r_min,
            R_max=r_max,
            box_size=box_size,
            mesh_shape=mesh_shape)

        return spherical_map

    return f


# ==========================================================
# Weak Lensing Born Approximation
# ==========================================================
def convergence_Born(cosmo,
                     density_planes,
                     r,
                     a,
                     z_source,
                     d_r,
                     dx=None,
                     coords=None):
    """
    Born approximation convergence for both spherical and flat geometries.

    Parameters
    ----------
    cosmo : jc.Cosmology
        Cosmology object
    density_planes : ndarray
        - Spherical: [n_planes, npix] - density on HEALPix grid
        - Flat: [n_planes, nx, ny] - density on Cartesian grid
        Note: d_R is already included in the density normalization
    r : ndarray
        Comoving distances to plane centers [n_planes]
    a : ndarray
        Scale factors at plane centers [n_planes]
    z_source : float or ndarray
        Source redshift(s)
    dx : float, optional
        Pixel size for flat-sky case (required for flat)
    coords : ndarray, optional
        Angular coordinates for flat-sky (required for flat)

    Returns
    -------
    convergence : ndarray
        Convergence map
    """
    # Constants
    # --- 1. Pre-computation and Shape Setup ---
    constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c)**2
    chi_s = jc.background.radial_comoving_distance(cosmo,
                                                   jc.utils.z2a(z_source))
    n_planes = len(r)

    # Detect geometry from input shape
    is_spherical = density_planes.ndim == 2  # [n_planes, npix]

    if not is_spherical:
        assert dx is not None and coords is not None, "dx and coords are required for flat geometry."

    # Reshape 1D arrays to [n_planes, 1, 1] for broadcasting with [n_planes, nx, ny]
    # Or to [n_planes, 1] for spherical geometry
    r_b = r.reshape(n_planes, *((1, ) * (density_planes.ndim - 1)))
    a_b = a.reshape(n_planes, *((1, ) * (density_planes.ndim - 1)))

    # --- 2. Vectorized Overdensity Calculation ---
    # Calculate mean density across spatial dimensions for each plane
    mean_axes = tuple(range(1, density_planes.ndim))
    rho_mean = jnp.mean(density_planes, axis=mean_axes, keepdims=True)
    # Avoid division by zero by adding a small epsilon where mean density is zero
    eps = jnp.finfo(rho_mean.dtype).eps
    safe_rho_mean = jnp.where(rho_mean == 0, eps, rho_mean)
    delta = density_planes / safe_rho_mean - 1

    # --- 3. Vectorized Lensing Kernel and Weighting ---
    # Combine all factors except interpolation
    # This includes the geometric term: dχ * χ / a(χ)
    kappa_contributions = delta * (d_r * r_b / a_b)
    kappa_contributions *= constant_factor
    # --- 4. Interpolation (for Flat-Sky only) ---
    if not is_spherical:
        # Define the interpolation function for a SINGLE plane
        def interpolate_plane(delta_plane, chi_plane):
            physical_coords = coords * chi_plane / dx
            return map_coordinates(delta_plane,
                                   physical_coords - 0.5,
                                   order=1,
                                   mode="wrap")

        # Use vmap to apply the function across all planes efficiently
        kappa_contributions = jax.vmap(interpolate_plane)(kappa_contributions,
                                                          r)

    # --- 5. Final Assembly ---
    # In case of multiple source redshifts, and a flat sky approximation,
    # We need to add a dimension to match the 2D shape of the kappa contributions
    if jnp.ndim(z_source) > 0 and not is_spherical:
        chi_s = jnp.expand_dims(chi_s, axis=1)
    # Apply the constant factor and the lensing efficiency kernel: (χs - χ) / χs
    lensing_efficiency = jnp.clip(1.0 - (r_b / chi_s), 0, 1000)
    # Add a dimension for broadcasting the redshift dimension
    lensing_efficiency = jnp.expand_dims(lensing_efficiency, axis=-1)
    kappa_contributions = jnp.expand_dims(kappa_contributions, axis=1)
    # Multiply the weighted delta by the lensing kernel and constant
    final_contributions = lensing_efficiency * kappa_contributions

    # Sum contributions from all planes to get the final convergence map
    # For multiple redshifts, preserve the redshift dimension
    convergence = jnp.sum(final_contributions, axis=0)

    # Handle single vs multiple redshift cases
    if jnp.ndim(z_source) == 0:  # Single redshift case
        convergence = jnp.squeeze(convergence, axis=0)

    return convergence
