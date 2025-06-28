import jax
import jax.numpy as jnp
import jax_cosmo
import jax_cosmo as jc
import jax_cosmo.constants as constants
from jax.scipy.ndimage import map_coordinates

from jaxpm.distributed import uniform_particles
from jaxpm.painting import cic_paint, cic_paint_2d, cic_paint_dx
from jaxpm.spherical import paint_spherical
from jaxpm.utils import gaussian_smoothing


def density_plane_fn(box_shape,
                     box_size,
                     density_plane_width,
                     density_plane_npix,
                     sharding=None):

    def f(t, y, args):
        positions = y[0]
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

        # Apply density normalization
        density_plane = density_plane / ((nx / density_plane_npix) *
                                         (ny / density_plane_npix) * w)
        return density_plane

    return f


def spherical_density_fn(box_shape,
                         box_size,
                         nside,
                         fov,
                         center_radec,
                         observer_position,
                         d_R,
                         sharding=None):

    def f(t, y, args):
        positions = y[0]
        nx, ny, nz = box_shape
        bx, by, bz = box_size
        cosmo = args
        # Converts time t to comoving distance in voxel coordinates
        w = d_R / box_size[2] * box_shape[2]
        center = ((jc.background.radial_comoving_distance(cosmo, t)) / bz) * nz

        # Apply sharding in order to recover sharding when taking gradients
        if sharding is not None:
            positions = jax.lax.with_sharding_constraint(positions, sharding)

        density_mesh = cic_paint_dx(positions)
        # Project to spherical map
        spherical_map = paint_spherical(density_mesh, nside, fov, center_radec,
                                        observer_position, box_size, center,
                                        d_R)
        return spherical_map

    return f


# ==========================================================
# Weak Lensing Born Approximation
# ==========================================================
def convergence_Born(cosmo, density_planes, r, a, dx, dz, coords, z_source):
    """
    Compute Born-approximation lensing convergence maps.

    Parameters
    ----------
    cosmo : jc.Cosmology
        Cosmology object.
    density_planes : ndarray
        3D array of lensing density planes [nx, ny, n_planes].
    r, a : ndarray
        Comoving distances and scale factors per plane.
    dx : float
        Pixel scale.
    dz : float
        Redshift bin width.
    coords : ndarray
        Angular coordinates grid [2, N, 2] in radians.
    z_source : ndarray
        Source redshifts.

    Returns
    -------
    convergence : ndarray
        2D convergence map for each source redshift.
    """
    constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c)**2
    # Compute comoving distance of source galaxies
    r_s = jc.background.radial_comoving_distance(cosmo, 1 / (1 + z_source))
    n_planes = len(r)

    def scan_fn(carry, i):
        density_planes, a, r = carry

        p = density_planes[:, :, i]
        density_normalization = dz * r[i] / a[i]
        p = (p - p.mean()) * constant_factor * density_normalization

        # Interpolate at the density plane coordinates
        im = map_coordinates(p, coords * r[i] / dx - 0.5, order=1, mode="wrap")

        return carry, im * jnp.clip(1.0 -
                                    (r[i] / r_s), 0, 1000).reshape([-1, 1, 1])

    _, convergence = jax.lax.scan(scan_fn, (density_planes, a, r),
                                  jnp.arange(n_planes))
    return convergence.sum(axis=0)


def spherical_convergence_Born(cosmo, density_planes, r, a, nside, z_source):
    """
    Compute Born-approximation lensing convergence maps on a sphere.

    Parameters
    ----------
    cosmo : jc.Cosmology
        Cosmology object.
    density_planes : ndarray
        3D array of lensing density planes [n_planes, npix].
    r, a : ndarray
        Comoving distances and scale factors per plane.
    nside : int
        Healpix nside parameter.
    z_source : ndarray
        Source redshifts.

    Returns
    -------
    convergence : ndarray
        2D convergence map for each source redshift.
    """
    constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c)**2
    # Compute comoving distance of source galaxies
    r_s = jc.background.radial_comoving_distance(cosmo, 1 / (1 + z_source))
    n_planes = len(r)

    def scan_fn(carry, i):
        density_planes, a, r = carry

        p = density_planes[i, :]
        density_normalization = r[i] / a[
            i]  # This normalization needs to be checked
        p = (p - p.mean()) * constant_factor * density_normalization

        return carry, p * jnp.clip(1.0 -
                                   (r[i] / r_s), 0, 1000).reshape([-1, 1])

    _, convergence = jax.lax.scan(scan_fn, (density_planes, a, r),
                                  jnp.arange(n_planes))
    return convergence.sum(axis=0)
