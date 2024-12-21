import jax
import pytest
from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve , RecursiveCheckpointAdjoint, BacksolveAdjoint
from helpers import MSE
from jax import numpy as jnp

from jaxpm.distributed import uniform_particles
from jaxpm.painting import cic_paint, cic_paint_dx
from jaxpm.pm import lpt, make_diffrax_ode


@pytest.mark.single_device
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("absolute_painting", [True, False])
@pytest.mark.parametrize("adjoint", ['DTO', 'OTD'])
def test_nbody_grad(simulation_config, initial_conditions, lpt_scale_factor,
                    nbody_from_lpt1, nbody_from_lpt2, cosmo, order,
                    absolute_painting , adjoint):

    mesh_shape, _ = simulation_config
    cosmo._workspace = {}

    if adjoint == 'OTD':
      pytest.skip("OTD adjoint not implemented yet (needs PFFT3D JVP)")

    adjoint = RecursiveCheckpointAdjoint() if adjoint == 'DTO' else BacksolveAdjoint(solver=Dopri5())

    @jax.jit
    @jax.grad
    def forward_model(initial_conditions, cosmo):

        # Initial displacement
        if absolute_painting:
            particles = uniform_particles(mesh_shape)
            dx, p, _ = lpt(cosmo,
                           initial_conditions,
                           particles,
                           a=lpt_scale_factor,
                           order=order)
            ode_fn = ODETerm(make_diffrax_ode(cosmo, mesh_shape))
            y0 = jnp.stack([particles + dx, p])

        else:
            dx, p, _ = lpt(cosmo,
                           initial_conditions,
                           a=lpt_scale_factor,
                           order=order)
            ode_fn = ODETerm(
                make_diffrax_ode(cosmo, mesh_shape, paint_absolute_pos=False))
            y0 = jnp.stack([dx, p])

        solver = Dopri5()
        controller = PIDController(rtol=1e-7,
                                   atol=1e-7,
                                   pcoeff=0.4,
                                   icoeff=1,
                                   dcoeff=0)

        saveat = SaveAt(t1=True)

        solutions = diffeqsolve(ode_fn,
                                solver,
                                t0=lpt_scale_factor,
                                t1=1.0,
                                dt0=None,
                                y0=y0,
                                adjoint=adjoint,
                                stepsize_controller=controller,
                                saveat=saveat)

        if absolute_painting:
            final_field = cic_paint(jnp.zeros(mesh_shape), solutions.ys[-1, 0])
        else:
            final_field = cic_paint_dx(solutions.ys[-1, 0])

        return MSE(final_field,
                   nbody_from_lpt1 if order == 1 else nbody_from_lpt2)

    bad_initial_conditions = initial_conditions + jax.random.normal(
        jax.random.PRNGKey(0), initial_conditions.shape) * 0.5
    best_ic = forward_model(initial_conditions, cosmo)
    bad_ic = forward_model(bad_initial_conditions, cosmo)

    assert jnp.max(best_ic) < 1e-5
    assert jnp.max(bad_ic) > 1e-5
