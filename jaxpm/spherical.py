import jax.numpy as jnp
import jax_healpy as jhp
import matplotlib.pyplot as plt
import jax
from functools import partial
import healpy as hp

@partial(jax.jit, static_argnames=('nside', 'fov', 'center_radec' , 'd_R' , 'box_size'))
def paint_spherical(volume, nside, fov, center_radec, observer_position, box_size, R, d_R):
    width, height, depth = volume.shape
    ra0, dec0 = center_radec
    fov_width, fov_height = fov

    pixel_scale_x = fov_width / width
    pixel_scale_y = fov_height / height

    res_deg = jhp.nside2resol(nside, arcmin=True) / 60
    if pixel_scale_x > res_deg or pixel_scale_y > res_deg:
        print(f"WARNING Pixel scale ({pixel_scale_x:.4f} deg, {pixel_scale_y:.4f} deg) is larger than the Healpy resolution ({res_deg:.4f} deg). Increase the field of view or decrease the nside.")

    y_idx, x_idx = jnp.indices((height, width))
    ra_grid = ra0 + x_idx * pixel_scale_x
    dec_grid = dec0 + y_idx * pixel_scale_y

    ra_flat = ra_grid.flatten() * jnp.pi / 180.0
    dec_flat = dec_grid.flatten() * jnp.pi / 180.0
    R_s = jnp.arange(0 , d_R, 1.0) + R

    XYZ = R_s.reshape(-1, 1, 1) * jhp.ang2vec(ra_flat, dec_flat, lonlat=False)
    observer_position = jnp.array(observer_position)
    # Convert observer position from box units to grid units
    observer_position = observer_position / jnp.array(box_size) *  jnp.array(volume.shape)
    
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
