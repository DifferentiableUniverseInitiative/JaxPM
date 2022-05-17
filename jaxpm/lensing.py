import jax 
import jax.numpy as jnp

from jaxpm.painting import cic_paint_2d

def density_plane(positions,
                  box_shape,
                  center,
                  width,
                  plane_resolution):
                  
    nx, ny, nz = box_shape
    xy = positions[..., :2]
    d = positions[..., 2]

    # Apply 2d periodic conditions
    xy = jnp.mod(xy, nx)

    # Rescaling positions to target grid
    xy = xy / nx * plane_resolution

    # Selecting only particles that fall inside the volume of interest
    mask = (d > (center - width / 2)) & (d <= (center + width / 2))

    # Painting density plane
    density_plane = cic_paint_2d(jnp.zeros([plane_resolution, plane_resolution]), xy[mask])

    # Apply density normalization
    density_plane = density_plane / ((nx / plane_resolution) *
                                     (ny / plane_resolution) * (width))

    return density_plane
