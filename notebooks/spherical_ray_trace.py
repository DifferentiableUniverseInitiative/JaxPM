
import os
os.environ["JC_CACHE"] = "off"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from functools import partial

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from diffrax import ODETerm, SaveAt, diffeqsolve , SemiImplicitEuler , ConstantStepSize
from jax_cosmo.scipy.integrate import simps
from jaxpm.pm import linear_field  ,lpt
from jaxpm.ode import symplectic_ode
from jaxpm.lensing import spherical_density_fn , spherical_convergence_Born
import matplotlib.pyplot as plt



jax.config.update("jax_enable_x64", False)


Planck18 = partial(
    jc.Cosmology,
    # Omega_m = 0.3111
    Omega_c=0.2607,
    Omega_b=0.0490,
    Omega_k=0.0,
    h=0.6766,
    n_s=0.9665,
    sigma8=0.8102,
    w0=-1.0,
    wa=0.0,
)

fiducial_cosmo = Planck18()



mesh_shape = (256, 256, 256)
box_size = (800.0, 800.0, 800.0)
observer_position = (400.0, 400.0, 400.0)  # Mpc/h
nside = 16
fov = (180., 360.) # degrees
center_radec = (0, 0) # degrees
d_R = 50. # Mpc/h
min_redshift = 0.0

from scipy.stats import norm

max_comoving_distance = box_size[2]  # in Mpc/h
max_redshift = (1 / jc.background.a_of_chi(fiducial_cosmo, max_comoving_distance / fiducial_cosmo.h) - 1).squeeze()
z = jnp.linspace(0, max_redshift, 1000)

nz_shear = [
    jc.redshift.kde_nz(
        z, norm.pdf(z, loc=z_center, scale=0.01), bw=0.01, zmax=max_redshift, gals_per_arcmin2=g
    )
    for z_center, g in zip([0.1, 0.2, 0.25, 0.3], [7, 8.5, 7.5, 7])
]
nbins = len(nz_shear)


# Plotting the redshift distribution
z = jnp.linspace(0, 1.2, 128)

for i in range(nbins):
    plt.plot(
        z,
        nz_shear[i](z) * nz_shear[i].gals_per_arcmin2,
        color=f"C{i}",
        label=f"Bin {i}",
    )
plt.savefig('redshift_distribution.png')



t0 = 0.1  # Initial scale factor
t1 = 1.0  # Final scale factor
dt0 = 0.05  # Initial time step

@jax.jit
def run_simulation(cosmo):
    # Create a small function to generate the matter power spectrum
    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(cosmo, k)
    pk_fn = lambda x: jnp.interp(x.reshape([-1]), k, pk).reshape(x.shape)

    # Create initial conditions
    initial_conditions = linear_field(mesh_shape, box_size, pk_fn, seed=jax.random.PRNGKey(0))    
    # Initial displacement
    dx, p, f = lpt(cosmo, initial_conditions, particles=None,a=0.1,order=1)
    
    # Evolve the simulation forward
    drift , kick = symplectic_ode(mesh_shape, paint_absolute_pos=False)
    ode_fn = ODETerm(kick), ODETerm(drift)
    solver = SemiImplicitEuler()


    a_init = t0
    n_lens = int((box_size[-1] - observer_position[-1]) // d_R)
    r = jnp.linspace(0.0, box_size[-1] - observer_position[-1], n_lens + 1)
    r_center = 0.5 * (r[1:] + r[:-1])
    a_center = jc.background.a_of_chi(cosmo, r_center)
    saveat = SaveAt(ts=a_center[::-1], fn=spherical_density_fn(
        mesh_shape, box_size, nside, observer_position, d_R
    ))

    stepsize_controller = ConstantStepSize()
    res = diffeqsolve(ode_fn,
                      solver,
                      t0=t0,
                      t1=t1,
                      dt0=dt0,
                      y0=(p , dx),
                      args=cosmo,
                      saveat=saveat,
                      stepsize_controller=stepsize_controller)

    density_planes = res.ys
    return initial_conditions ,  dx , density_planes , res.stats

initial_conditions , lpt_displacements , density_planes , solver_stats = run_simulation(fiducial_cosmo)


import healpy as hp
fig = plt.figure(figsize=(16, 10))
for i , density_plane in enumerate(density_planes):
    hp.mollview(
        density_plane,
        title=f"Density Plane {i}",
        cmap="viridis",
        sub=(4 , 4 , i + 1),
        bgcolor=(0,)*4,
        cbar=False)

plt.tight_layout()
plt.savefig('density_planes.png')

def compute_spherical_convergence(density_planes, cosmo, nside, min_redshift=0.0 , max_redshift=2.0):
    n_lens = int((box_size[-1] - observer_position[-1]) // d_R)
    r = jnp.linspace(0.0, box_size[-1] - observer_position[-1], n_lens + 1)
    r_center = 0.5 * (r[1:] + r[:-1])
    a_center = jc.background.a_of_chi(cosmo, r_center)

    lightcone = density_planes

    lightcone = lightcone[::-1]
    # lightcone = jnp.transpose(lightcone, axes=(1, 2, 0)) # Not needed for spherical maps

    convergence_maps = [
                simps(
                    lambda z: nz(z).reshape([-1, 1])
                    * spherical_convergence_Born(fiducial_cosmo, lightcone, r_center, a_center, nside, z),
                    min_redshift,
                    max_redshift,
                    N=32,
                )
                for nz in nz_shear
            ]
    return convergence_maps

kappas = compute_spherical_convergence(density_planes, fiducial_cosmo, nside)

fig = plt.figure(figsize=(16, 10))
for i, kappa in enumerate(kappas):
    hp.mollview(
        kappa,
        title=f"Convergence Map {i}",
        cmap="viridis",
        sub=(2, 2, i + 1),
        bgcolor=(0,)*4,
        cbar=False)

plt.tight_layout()
plt.savefig('convergence_maps.png')

import numpy as np
import os
os.makedirs('experiments/results', exist_ok=True)

print("Saving convergence maps...")
for i, kappa in enumerate(kappas):
    filename = f'experiments/results/convergence_map_{i}_physical.npy'
    np.save(filename, np.array(kappa))
    print(f"Saved convergence map {i} to {filename}")

print(f"Generated {len(kappas)} convergence maps with shape {kappas[0].shape} (nside={nside})")