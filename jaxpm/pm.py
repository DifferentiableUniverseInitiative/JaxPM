import warnings

import jax.numpy as jnp
import jax_cosmo as jc

from jaxpm.distributed import fft3d, ifft3d, normal_field
from jaxpm.growth import (dGf2a, dGfa, growth_factor, growth_factor_second,
                          growth_rate, growth_rate_second)
from jaxpm.kernels import (PGD_kernel, compensation_kernel, fftk,
                           gradient_kernel, invlaplace_kernel,
                           longrange_kernel)
from jaxpm.painting import paint, readout


def pm_forces(positions,
              mesh_shape=None,
              delta=None,
              r_split=0,
              paint_absolute_pos=None,
              halo_size=0,
              sharding=None,
              order='CIC',
              deconvolution=False,
              initial_particles=None):
    """
    Computes gravitational forces on particles using a PM scheme.

    Parameters
    ----------
    order : int or str
        Mass-assignment order forwarded to ``paint``/``readout``
        (NGP=1, CIC=2, TSC=3, PCS=4).
    deconvolution : bool
        Deconvolve the assignment window from the painted density. Applied in
        Fourier space (we are already in k-space for the PM kernels), which
        avoids the FFT/iFFT round-trip of a real-space ``paint(deconvolution=True)``.
    initial_particles : {None, 'uniform'}
        ``None`` -> ``positions`` are absolute, ``'uniform'`` -> ``positions``
        are displacements from a uniform grid.
    paint_absolute_pos : bool, optional
        Deprecated -- use ``initial_particles`` instead (``True`` -> ``None``,
        ``False`` -> ``'uniform'``).
    """
    if mesh_shape is None:
        assert (delta is not None),\
          "If mesh_shape is not provided, delta should be provided"
        mesh_shape = delta.shape

    if paint_absolute_pos is not None:
        warnings.warn(
            "paint_absolute_pos is deprecated; use initial_particles instead "
            "(None for absolute positions, 'uniform' for displacements).",
            DeprecationWarning,
            stacklevel=2)
        initial_particles = None if paint_absolute_pos else 'uniform'

    # Deconvolution are done opportunistically in Fourier space, so we can avoid the extra FFT/iFFT round-trip
    paint_fn = lambda pos: paint(pos,
                                 initial_particles=initial_particles,
                                 order=order,
                                 halo_size=halo_size,
                                 sharding=sharding,
                                 deconvolution=False)
    read_fn = lambda grid_mesh, pos: readout(grid_mesh,
                                             pos,
                                             initial_particles=
                                             initial_particles,
                                             order=order,
                                             halo_size=halo_size,
                                             sharding=sharding)

    if delta is None:
        field = paint_fn(positions)
        delta_k = fft3d(field)
    elif jnp.isrealobj(delta):
        delta_k = fft3d(delta)
    else:
        delta_k = delta

    kvec = fftk(delta_k)
    # Deconvolve the assignment window directly in Fourier space (one multiply,
    # no extra FFT/iFFT) -- only meaningful for the density we just painted.
    if deconvolution and delta is None:
        delta_k = delta_k * compensation_kernel(kvec, order)
    # Computes gravitational potential
    pot_k = delta_k * invlaplace_kernel(kvec) * longrange_kernel(
        kvec, r_split=r_split)
    # Computes gravitational forces
    forces = jnp.stack([
        read_fn(ifft3d(-gradient_kernel(kvec, i) * pot_k),positions
        ) for i in range(3)], axis=-1) # yapf: disable

    return forces


def lpt(cosmo,
        initial_conditions,
        particles=None,
        a=0.1,
        halo_size=0,
        sharding=None,
        order=1):
    """
    Computes first and second order LPT displacement and momentum,
    e.g. Eq. 2 and 3 [Jenkins2010](https://arxiv.org/pdf/0910.0258)
    """
    initial_particles = None if particles is not None else 'uniform'
    if particles is None:
        particles = jnp.zeros_like(initial_conditions,
                                   shape=(*initial_conditions.shape, 3))

    a = jnp.atleast_1d(a)
    E = jnp.sqrt(jc.background.Esqr(cosmo, a))
    delta_k = fft3d(initial_conditions)
    initial_force = pm_forces(particles,
                              delta=delta_k,
                              initial_particles=initial_particles,
                              halo_size=halo_size,
                              sharding=sharding)
    dx = growth_factor(cosmo, a) * initial_force
    p = a**2 * growth_rate(cosmo, a) * E * dx
    f = a**2 * E * dGfa(cosmo, a) * initial_force
    if order == 2:
        kvec = fftk(delta_k)
        pot_k = delta_k * invlaplace_kernel(kvec)

        delta2 = 0
        shear_acc = 0
        # for i, ki in enumerate(kvec):
        for i in range(3):
            # Add products of diagonal terms = 0 + s11*s00 + s22*(s11+s00)...
            # shear_ii = jnp.fft.irfftn(- ki**2 * pot_k)
            nabla_i_nabla_i = gradient_kernel(kvec, i)**2
            shear_ii = ifft3d(nabla_i_nabla_i * pot_k)
            delta2 += shear_ii * shear_acc
            shear_acc += shear_ii

            # for kj in kvec[i+1:]:
            for j in range(i + 1, 3):
                # Substract squared strict-up-triangle terms
                # delta2 -= jnp.fft.irfftn(- ki * kj * pot_k)**2
                nabla_i_nabla_j = gradient_kernel(kvec, i) * gradient_kernel(
                    kvec, j)
                delta2 -= ifft3d(nabla_i_nabla_j * pot_k)**2

        delta_k2 = fft3d(delta2)
        init_force2 = pm_forces(particles,
                                delta=delta_k2,
                                initial_particles=initial_particles,
                                halo_size=halo_size,
                                sharding=sharding)
        # NOTE: growth_factor_second is renormalized: - D2 = 3/7 * growth_factor_second
        dx2 = 3 / 7 * growth_factor_second(cosmo, a) * init_force2
        p2 = a**2 * growth_rate_second(cosmo, a) * E * dx2
        f2 = a**2 * E * dGf2a(cosmo, a) * init_force2

        dx += dx2
        p += p2
        f += f2

    return dx, p, f


def linear_field(mesh_shape, box_size, pk, seed, sharding=None):
    """
    Generate initial conditions.
    """
    # Initialize a random field with one slice on each gpu
    field = normal_field(seed=seed, shape=mesh_shape, sharding=sharding)
    field = fft3d(field)
    kvec = fftk(field)
    kmesh = sum((kk / box_size[i] * mesh_shape[i])**2
                for i, kk in enumerate(kvec))**0.5
    pkmesh = pk(kmesh) * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]) / (
        box_size[0] * box_size[1] * box_size[2])

    field = field * jnp.sqrt(pkmesh)
    field = ifft3d(field)
    return field


def pgd_correction(pos, mesh_shape, params, order='CIC', deconvolution=False):
    """
    improve the short-range interactions of PM-Nbody simulations with potential gradient descent method,
    based on https://arxiv.org/abs/1804.00671

    args:
      pos: particle positions [npart, 3]
      params: [alpha, kl, ks] pgd parameters
      order: mass-assignment order for paint/readout (NGP/CIC/TSC/PCS)
      deconvolution: deconvolve the assignment window (in Fourier space)
    """
    delta = paint(pos, grid_mesh=jnp.zeros(mesh_shape), order=order)
    delta_k = fft3d(delta)
    kvec = fftk(delta_k)
    if deconvolution:
        delta_k = delta_k * compensation_kernel(kvec, order)
    alpha, kl, ks = params
    PGD_range = PGD_kernel(kvec, kl, ks)

    pot_k_pgd = (delta_k * invlaplace_kernel(kvec)) * PGD_range

    forces_pgd = jnp.stack([
        readout(fft3d(-gradient_kernel(kvec, i) * pot_k_pgd), pos, order=order)
        for i in range(3)
    ],
                           axis=-1)

    dpos_pgd = forces_pgd * alpha

    return dpos_pgd
