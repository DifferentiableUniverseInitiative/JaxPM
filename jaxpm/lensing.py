import jax
import jax.numpy as jnp
import jax_cosmo
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
        pixel_area = (box_size[0] / density_plane_npix) * (box_size[1] / density_plane_npix)
        shell_thickness_physical = density_plane_width  # Already in physical units
        pixel_volume = pixel_area * shell_thickness_physical

        # Convert counts to density (particles per unit volume)
        density_plane = density_plane / pixel_volume

        return density_plane

    return f


def spherical_density_fn(mesh_shape, box_size, nside, observer_position, d_R):

    def f(t, y, args):
        positions = y
        cosmo = args

        positions = uniform_particles(mesh_shape) + positions

        # Calculate comoving distance range for this shell
        r_center = jc.background.radial_comoving_distance(cosmo, t)
        r_max = jnp.clip(r_center + d_R / 2, 0, box_size[2])
        r_min = jnp.clip(r_center - d_R / 2, 0, box_size[2])

        # Paint particles in this shell onto a HEALPix map
        spherical_map = paint_particles_spherical(
            positions,
            nside=nside,
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
def convergence_Born(cosmo, density_planes, r, a, z_source, d_r,
                     dx=None, coords=None):
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
    constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c)**2
    chi_s = jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(z_source)) 
    n_planes = len(r)
    
    # Detect geometry from input shape
    is_spherical = density_planes.ndim == 2  # [n_planes, npix]
    
    if not is_spherical:
        assert dx is not None and coords is not None, "dx and coords required for flat geometry"
    
    def scan_fn(carry, i):
        density_planes, a, r = carry
        
        # Get density for this plane - plane index is always first
        rho = density_planes[i]  # Works for both geometries
        
        # Convert density to overdensity: δ = ρ/ρ̄ - 1
        # Use proper cosmological mean density for correct normalization
        # The simulation density should be normalized to the mean density
        rho_mean = jnp.mean(rho)
        delta = rho / rho_mean - 1
 
        # Multiply by dχ * χ / a(χ) 
        chi = r[i]
        delta *= d_r * chi / a[i]
        # Multiply by the constant factor
        delta *= constant_factor
        if not is_spherical:
            # For flat-sky: interpolate at light ray positions
            physical_coords = coords * chi / dx
            delta = map_coordinates(
                delta, 
                physical_coords - 0.5,
                order=1, 
                mode="wrap"
            )

        # Lensing kernel: W(χ,χs)/a(χ)
        # Already multiplied by dχ * χ / a(χ) above
        # So we need to multiply by  χs - χ / χs
        
        lensing_efficiency = jnp.clip(1.0 - (chi / chi_s), 0, 1000)
        lensing_efficiency = lensing_efficiency.reshape(-1 , *(1,) * delta.ndim)
        # Apply kernel to overdensity
        kappa_contribution = lensing_efficiency * delta 
        return carry, kappa_contribution.sum(axis=0)
    
    # Integrate over all planes
    _, kappa_contributions = jax.lax.scan(scan_fn, (density_planes, a, r), 
                                          jnp.arange(n_planes))
    
    # Sum contributions
    convergence = jnp.sum(kappa_contributions, axis=0)
    
    return convergence


