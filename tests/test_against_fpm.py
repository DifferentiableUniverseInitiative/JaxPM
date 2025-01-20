import pytest
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve
from helpers import MSE, MSRE
from jax import numpy as jnp

from jaxdecomp import ShardedArray
from jaxpm.distributed import uniform_particles
from jaxpm.painting import cic_paint, cic_paint_dx
from jaxpm.pm import lpt, make_diffrax_ode
from jaxpm.utils import power_spectrum
import jax 
_TOLERANCE = 1e-4
_PM_TOLERANCE = 1e-3


@pytest.mark.single_device
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("shardedArrayAPI", [True, False])
def test_lpt_absolute(simulation_config, initial_conditions, lpt_scale_factor,
                      fpm_lpt1_field, fpm_lpt2_field, cosmo, order , shardedArrayAPI):

    mesh_shape, box_shape = simulation_config
    cosmo._workspace = {}
    particles = uniform_particles(mesh_shape)

    if shardedArrayAPI:
        particles = ShardedArray(particles)
        initial_conditions = ShardedArray(initial_conditions)

    # Initial displacement
    dx, _, _ = lpt(cosmo,
                   initial_conditions,
                   particles,
                   a=lpt_scale_factor,
                   order=order)

    fpm_ref_field = fpm_lpt1_field if order == 1 else fpm_lpt2_field

    lpt_field = cic_paint(jnp.zeros(mesh_shape), particles + dx)
    lpt_field_arr, = jax.tree.leaves(lpt_field)
    _, jpm_ps = power_spectrum(lpt_field_arr, box_shape=box_shape)
    _, fpm_ps = power_spectrum(fpm_ref_field, box_shape=box_shape)

    assert MSE(lpt_field_arr, fpm_ref_field) < _TOLERANCE
    assert MSRE(jpm_ps, fpm_ps) < _TOLERANCE

    if shardedArrayAPI:
        assert type(dx) == ShardedArray
        assert type(lpt_field) == ShardedArray


@pytest.mark.single_device
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("shardedArrayAPI", [True, False])
def test_lpt_relative(simulation_config, initial_conditions, lpt_scale_factor,
                      fpm_lpt1_field, fpm_lpt2_field, cosmo, order , shardedArrayAPI):

    mesh_shape, box_shape = simulation_config
    cosmo._workspace = {}
    if shardedArrayAPI:
        initial_conditions = ShardedArray(initial_conditions)

    # Initial displacement
    dx, _, _ = lpt(cosmo, initial_conditions, a=lpt_scale_factor, order=order)

    lpt_field = cic_paint_dx(dx)

    fpm_ref_field = fpm_lpt1_field if order == 1 else fpm_lpt2_field
    lpt_field_arr, = jax.tree.leaves(lpt_field)
    _, jpm_ps = power_spectrum(lpt_field_arr, box_shape=box_shape)
    _, fpm_ps = power_spectrum(fpm_ref_field, box_shape=box_shape)

    assert MSE(lpt_field_arr, fpm_ref_field) < _TOLERANCE
    assert MSRE(jpm_ps, fpm_ps) < _TOLERANCE

    if shardedArrayAPI:
        assert type(dx) == ShardedArray
        assert type(lpt_field) == ShardedArray

@pytest.mark.single_device
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("shardedArrayAPI", [True, False])
def test_nbody_absolute(simulation_config, initial_conditions,
                        lpt_scale_factor, nbody_from_lpt1, nbody_from_lpt2,
                        cosmo, order , shardedArrayAPI):

    mesh_shape, box_shape = simulation_config
    cosmo._workspace = {}
    particles = uniform_particles(mesh_shape)

    if shardedArrayAPI:
        particles = ShardedArray(particles)
        initial_conditions = ShardedArray(initial_conditions)

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

    y0 = jax.tree.map(lambda particles , dx , p : jnp.stack([particles + dx, p]), particles ,  dx, p)

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

    final_field_arr, = jax.tree.leaves(final_field)
    _, jpm_ps = power_spectrum(final_field_arr, box_shape=box_shape)
    _, fpm_ps = power_spectrum(fpm_ref_field, box_shape=box_shape)

    assert MSE(final_field_arr, fpm_ref_field) < _PM_TOLERANCE
    assert MSRE(jpm_ps, fpm_ps) < _PM_TOLERANCE

    if shardedArrayAPI:
        assert type(dx) == ShardedArray
        assert type( solutions.ys[-1, 0]) == ShardedArray
        assert type(final_field) == ShardedArray


@pytest.mark.single_device
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("shardedArrayAPI", [True, False])
def test_nbody_relative(simulation_config, initial_conditions,
                        lpt_scale_factor, nbody_from_lpt1, nbody_from_lpt2,
                        cosmo, order , shardedArrayAPI):

    mesh_shape, box_shape = simulation_config
    cosmo._workspace = {}

    if shardedArrayAPI:
        initial_conditions = ShardedArray(initial_conditions)

    # Initial displacement
    dx, p, _ = lpt(cosmo, initial_conditions, a=lpt_scale_factor, order=order)

    ode_fn = ODETerm(
        make_diffrax_ode(mesh_shape, paint_absolute_pos=False))

    solver = Dopri5()
    controller = PIDController(rtol=1e-9,
                               atol=1e-9,
                               pcoeff=0.4,
                               icoeff=1,
                               dcoeff=0)

    saveat = SaveAt(t1=True)

    y0 = jax.tree.map(lambda dx , p : jnp.stack([dx, p]), dx, p)

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

    final_field_arr, = jax.tree.leaves(final_field)
    _, jpm_ps = power_spectrum(final_field_arr, box_shape=box_shape)
    _, fpm_ps = power_spectrum(fpm_ref_field, box_shape=box_shape)

    assert MSE(final_field_arr, fpm_ref_field) < _PM_TOLERANCE
    assert MSRE(jpm_ps, fpm_ps) < _PM_TOLERANCE

    if shardedArrayAPI:
        assert type(dx) == ShardedArray
        assert type( solutions.ys[-1, 0]) == ShardedArray
        assert type(final_field) == ShardedArray
