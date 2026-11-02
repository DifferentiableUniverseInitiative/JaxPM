import jax.numpy as jnp
import jax_cosmo as jc

from jaxpm.distributed import fft3d
from jaxpm.growth import E, Gf, dGfa, gp
from jaxpm.growth import growth_factor as Gp
from jaxpm.kernels import (fftk, gradient_kernel, invlaplace_kernel,
                           longrange_kernel)
from jaxpm.painting import paint, readout
from jaxpm.pm import pm_forces


def symplectic_fpm_ode(
        mesh_shape,
        dt0,
        cosmo,
        paint_absolute_pos=None,  # deprecated, use initial_particles
        halo_size=0,
        sharding=None,
        order='CIC',
        deconvolution=False,
        initial_particles=None):

    def drift(a, vel, args):
        """
        state is a tuple (position, velocities)
        """
        t0 = a
        t1 = a + dt0
        ai = t0
        ac = (t0 * t1)**0.5
        af = t1

        drift_contr = (Gp(cosmo, af) - Gp(cosmo, ai)) / gp(cosmo, ac)
        dpos = 1 / (ac**3 * E(cosmo, ac)) * vel

        return dpos * (drift_contr / dt0)

    def kick(a, pos, args):
        """
        state is a tuple (position, velocities)
        """
        t0 = a
        t1 = t0 + dt0
        t2 = t1 + dt0
        t0t1 = (t0 * t1)**0.5
        t1t2 = (t1 * t2)**0.5
        ac = t1

        forces = (pm_forces(
            pos,
            mesh_shape=mesh_shape,
            paint_absolute_pos=paint_absolute_pos,
            halo_size=halo_size,
            sharding=sharding,
            order=order,
            deconvolution=deconvolution,
            initial_particles=initial_particles,
        ) * 1.5 * cosmo.Omega_m)

        dvel = 1.0 / (ac**2 * E(cosmo, ac)) * forces
        kick_factor_1 = (Gf(cosmo, t1) - Gf(cosmo, t0t1)) / dGfa(cosmo, t1)
        kick_factor_2 = (Gf(cosmo, t1t2) - Gf(cosmo, t1)) / dGfa(cosmo, t1)

        return dvel * ((kick_factor_1 + kick_factor_2) / dt0)

    def first_kick(a, pos, args):
        """
        state is a tuple (position, velocities)
        """
        cosmo = args
        t0 = a
        t1 = t0 + dt0
        t0t1 = (t0 * t1)**0.5

        forces = (pm_forces(
            pos,
            mesh_shape=mesh_shape,
            paint_absolute_pos=paint_absolute_pos,
            halo_size=halo_size,
            sharding=sharding,
            order=order,
            deconvolution=deconvolution,
            initial_particles=initial_particles,
        ) * 1.5 * cosmo.Omega_m)

        dvel = 1.0 / (a**2 * E(cosmo, a)) * forces
        kick_factor = (Gf(cosmo, t0t1) - Gf(cosmo, t0)) / dGfa(cosmo, t0)

        return dvel * (kick_factor / dt0)

    return drift, kick, first_kick


def symplectic_ode(
        mesh_shape,
        cosmo,
        paint_absolute_pos=None,  # deprecated, use initial_particles
        halo_size=0,
        sharding=None,
        order='CIC',
        deconvolution=False,
        initial_particles=None):

    def drift(a, vel, args):
        """
        state is a tuple (position, velocities)
        """
        # Computes the update of position (drift)
        dpos = 1 / (a**3 * E(cosmo, a)) * vel

        return dpos

    def kick(a, pos, args):
        """
        state is a tuple (position, velocities)
        """
        # Computes the update of velocity (kick)

        forces = (pm_forces(
            pos,
            mesh_shape=mesh_shape,
            paint_absolute_pos=paint_absolute_pos,
            halo_size=halo_size,
            sharding=sharding,
            order=order,
            deconvolution=deconvolution,
            initial_particles=initial_particles,
        ) * 1.5 * cosmo.Omega_m)

        # Computes the update of velocity (kick)
        dvel = 1.0 / (a**2 * E(cosmo, a)) * forces

        return dvel

    return drift, kick


def make_ode_fn(
        mesh_shape,
        paint_absolute_pos=None,  # deprecated, use initial_particles
        halo_size=0,
        sharding=None,
        order='CIC',
        deconvolution=False,
        initial_particles=None):

    def nbody_ode(state, a, cosmo):
        """
        state is a tuple (position, velocities)
        """
        pos, vel = state

        forces = pm_forces(
            pos,
            mesh_shape=mesh_shape,
            paint_absolute_pos=paint_absolute_pos,
            halo_size=halo_size,
            sharding=sharding,
            order=order,
            deconvolution=deconvolution,
            initial_particles=initial_particles) * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

        # Computes the update of velocity (kick)
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

        return dpos, dvel

    return nbody_ode


def make_diffrax_ode(
        mesh_shape,
        paint_absolute_pos=None,  # deprecated, use initial_particles
        halo_size=0,
        sharding=None,
        order='CIC',
        deconvolution=False,
        initial_particles=None):

    def nbody_ode(a, state, args):
        """
        state is a tuple (position, velocities)
        """
        pos, vel = state
        cosmo = args

        forces = pm_forces(
            pos,
            mesh_shape=mesh_shape,
            paint_absolute_pos=paint_absolute_pos,
            halo_size=halo_size,
            sharding=sharding,
            order=order,
            deconvolution=deconvolution,
            initial_particles=initial_particles) * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

        # Computes the update of velocity (kick)
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

        return jnp.stack([dpos, dvel])

    return nbody_ode


def make_neural_ode_fn(model, mesh_shape):

    def neural_nbody_ode(state, a, cosmo: jc.Cosmology, params):
        """
        state is a tuple (position, velocities)
        """
        pos, vel = state
        delta = paint(pos, grid_mesh=jnp.zeros(mesh_shape), order='cic')
        delta_k = fft3d(delta)
        kvec = fftk(delta_k)

        # Computes gravitational potential
        pot_k = delta_k * invlaplace_kernel(kvec) * longrange_kernel(kvec,
                                                                     r_split=0)

        # Apply a correction filter
        kk = jnp.sqrt(sum((ki / jnp.pi)**2 for ki in kvec))
        pot_k = pot_k * (1. + model.apply(params, kk, jnp.atleast_1d(a)))

        # Computes gravitational forces
        forces = jnp.stack([
            readout(fft3d(-gradient_kernel(kvec, i) * pot_k), pos, order='cic')
            for i in range(3)
        ],
                           axis=-1)

        forces = forces * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

        # Computes the update of velocity (kick)
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

        return dpos, dvel

    return neural_nbody_ode
