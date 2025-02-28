import pytest
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve
from helpers import MSE, MSRE
from jax import numpy as jnp

from jaxpm.distributed import uniform_particles
from jaxpm.painting import cic_paint, cic_paint_dx
from jaxpm.pm import lpt, make_diffrax_ode
from jaxpm.utils import power_spectrum

_TOLERANCE = 1e-4
_PM_TOLERANCE = 1e-3


@pytest.mark.single_device
@pytest.mark.parametrize("order", [1, 2])
def test_lpt_absolute(simulation_config, initial_conditions, lpt_scale_factor,
                      fpm_lpt1_field, fpm_lpt2_field, cosmo, order):

    mesh_shape, box_shape = simulation_config
    cosmo._workspace = {}
    particles = uniform_particles(mesh_shape)

    # Initial displacement
    dx, _, _ = lpt(cosmo,
                   initial_conditions,
                   particles,
                   a=lpt_scale_factor,
                   order=order)

    fpm_ref_field = fpm_lpt1_field if order == 1 else fpm_lpt2_field

    lpt_field = cic_paint(jnp.zeros(mesh_shape), particles + dx)
    _, jpm_ps = power_spectrum(lpt_field, box_shape=box_shape)
    _, fpm_ps = power_spectrum(fpm_ref_field, box_shape=box_shape)

    assert MSE(lpt_field, fpm_ref_field) < _TOLERANCE
    assert MSRE(jpm_ps, fpm_ps) < _TOLERANCE


@pytest.mark.single_device
@pytest.mark.parametrize("order", [1, 2])
def test_lpt_relative(simulation_config, initial_conditions, lpt_scale_factor,
                      fpm_lpt1_field, fpm_lpt2_field, cosmo, order):

    mesh_shape, box_shape = simulation_config
    cosmo._workspace = {}
    # Initial displacement
    dx, _, _ = lpt(cosmo, initial_conditions, a=lpt_scale_factor, order=order)

    lpt_field = cic_paint_dx(dx)

    fpm_ref_field = fpm_lpt1_field if order == 1 else fpm_lpt2_field

    _, jpm_ps = power_spectrum(lpt_field, box_shape=box_shape)
    _, fpm_ps = power_spectrum(fpm_ref_field, box_shape=box_shape)

    assert MSE(lpt_field, fpm_ref_field) < _TOLERANCE
    assert MSRE(jpm_ps, fpm_ps) < _TOLERANCE


@pytest.mark.single_device
@pytest.mark.parametrize("order", [1, 2])
def test_nbody_absolute(simulation_config, initial_conditions,
                        lpt_scale_factor, nbody_from_lpt1, nbody_from_lpt2,
                        cosmo, order):

    mesh_shape, box_shape = simulation_config
    cosmo._workspace = {}
    particles = uniform_particles(mesh_shape)

    # Initial displacement
    dx, p, _ = lpt(cosmo,
                   initial_conditions,
                   particles,
                   a=lpt_scale_factor,
                   order=order)

    ode_fn = ODETerm(make_diffrax_ode(mesh_shape))

    solver = Dopri5()
    controller = PIDController(rtol=1e-8,
                               atol=1e-8,
                               pcoeff=0.4,
                               icoeff=1,
                               dcoeff=0)

    saveat = SaveAt(t1=True)

    y0 = jnp.stack([particles + dx, p])

    solutions = diffeqsolve(ode_fn,
                            solver,
                            t0=lpt_scale_factor,
                            t1=1.0,
                            dt0=None,
                            y0=y0,
                            args=cosmo,
                            stepsize_controller=controller,
                            saveat=saveat)

    final_field = cic_paint(jnp.zeros(mesh_shape), solutions.ys[-1, 0])

    fpm_ref_field = nbody_from_lpt1 if order == 1 else nbody_from_lpt2

    _, jpm_ps = power_spectrum(final_field, box_shape=box_shape)
    _, fpm_ps = power_spectrum(fpm_ref_field, box_shape=box_shape)

    assert MSE(final_field, fpm_ref_field) < _PM_TOLERANCE
    assert MSRE(jpm_ps, fpm_ps) < _PM_TOLERANCE


@pytest.mark.single_device
@pytest.mark.parametrize("order", [1, 2])
def test_nbody_relative(simulation_config, initial_conditions,
                        lpt_scale_factor, nbody_from_lpt1, nbody_from_lpt2,
                        cosmo, order):

    mesh_shape, box_shape = simulation_config
    cosmo._workspace = {}

    # Initial displacement
    dx, p, _ = lpt(cosmo, initial_conditions, a=lpt_scale_factor, order=order)

    ode_fn = ODETerm(make_diffrax_ode(mesh_shape, paint_absolute_pos=False))

    solver = Dopri5()
    controller = PIDController(rtol=1e-9,
                               atol=1e-9,
                               pcoeff=0.4,
                               icoeff=1,
                               dcoeff=0)

    saveat = SaveAt(t1=True)

    y0 = jnp.stack([dx, p])

    solutions = diffeqsolve(ode_fn,
                            solver,
                            t0=lpt_scale_factor,
                            t1=1.0,
                            dt0=None,
                            y0=y0,
                            args=cosmo,
                            stepsize_controller=controller,
                            saveat=saveat)

    final_field = cic_paint_dx(solutions.ys[-1, 0])

    fpm_ref_field = nbody_from_lpt1 if order == 1 else nbody_from_lpt2

    _, jpm_ps = power_spectrum(final_field, box_shape=box_shape)
    _, fpm_ps = power_spectrum(fpm_ref_field, box_shape=box_shape)

    assert MSE(final_field, fpm_ref_field) < _PM_TOLERANCE
    assert MSRE(jpm_ps, fpm_ps) < _PM_TOLERANCE
