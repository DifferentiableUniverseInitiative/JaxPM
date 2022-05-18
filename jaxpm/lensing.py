import jax 
import jax.numpy as jnp
import jax_cosmo.constants as constants
import jax_cosmo

from jax.scipy.ndimage import map_coordinates
from jaxpm.utils import gaussian_smoothing
from jaxpm.painting import cic_paint_2d

def density_plane(positions,
                  box_shape,
                  center,
                  width,
                  plane_resolution,
                  smoothing_sigma=None):
    """ Extacts a density plane from the simulation
    """
    nx, ny, nz = box_shape
    xy = positions[..., :2]
    d = positions[..., 2]

    # Apply 2d periodic conditions
    xy = jnp.mod(xy, nx)

    # Rescaling positions to target grid
    xy = xy / nx * plane_resolution

    # Selecting only particles that fall inside the volume of interest
    weight = jnp.where((d > (center - width / 2)) & (d <= (center + width / 2)), 1., 0.)
    # Painting density plane
    density_plane = cic_paint_2d(jnp.zeros([plane_resolution, plane_resolution]), xy, weight)

    # Apply density normalization
    density_plane = density_plane / ((nx / plane_resolution) *
                                     (ny / plane_resolution) * (width))

    # Apply Gaussian smoothing if requested
    if smoothing_sigma is not None:
        density_plane = gaussian_smoothing(density_plane, 
                                           smoothing_sigma)

    return density_plane


def convergence_Born(cosmo,
                     density_planes,
                     coords,
                     z_source):
  """
  Compute the Born convergence
  Args:
    cosmo: `Cosmology`, cosmology object.
    density_planes: list of dictionaries (r, a, density_plane, dx, dz), lens planes to use 
    coords: a 3-D array of angular coordinates in radians of N points with shape [batch, N, 2].
    z_source: 1-D `Tensor` of source redshifts with shape [Nz] .
    name: `string`, name of the operation.
  Returns:
    `Tensor` of shape [batch_size, N, Nz], of convergence values.
  """
  # Compute constant prefactor:
  constant_factor = 3 / 2 * cosmo.Omega_m * (constants.H0 / constants.c)**2
  # Compute comoving distance of source galaxies
  r_s = jax_cosmo.background.radial_comoving_distance(cosmo, 1 / (1 + z_source))

  convergence = 0
  for entry in density_planes:
    r = entry['r']; a = entry['a']; p = entry['plane']
    dx = entry['dx']; dz = entry['dz']
    # Normalize density planes
    density_normalization = dz * r / a
    p = (p - p.mean()) * constant_factor * density_normalization

    # Interpolate at the density plane coordinates
    im = map_coordinates(p, 
                         coords * r / dx - 0.5, 
                         order=1, mode="wrap")

    convergence += im * jnp.clip(1. - (r / r_s), 0, 1000).reshape([-1, 1, 1])

  return convergence
