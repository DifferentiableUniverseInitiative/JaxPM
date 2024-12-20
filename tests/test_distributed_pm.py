from conftest import initialize_distributed

initialize_distributed()  # ignore : E402

import jax  # noqa : E402
import jax.numpy as jnp  # noqa : E402
import pytest  # noqa : E402
from diffrax import SaveAt  # noqa : E402
from diffrax import Dopri5, ODETerm, PIDController, diffeqsolve
from helpers import MSE  # noqa : E402
from jax import lax  # noqa : E402
from jax.experimental.multihost_utils import process_allgather  # noqa : E402
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P  # noqa : E402

from jaxpm.distributed import uniform_particles  # noqa : E402
from jaxpm.painting import cic_paint, cic_paint_dx  # noqa : E402
from jaxpm.pm import lpt, make_diffrax_ode  # noqa : E402

_TOLERANCE = 3.0  # 🙃🙃


@pytest.mark.distributed
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("absolute_painting", [True, False])
def test_distrubted_pm(simulation_config, initial_conditions, cosmo, order,
                       absolute_painting):

    mesh_shape, box_shape = simulation_config
    # SINGLE DEVICE RUN
    cosmo._workspace = {}
    if absolute_painting:
        particles = uniform_particles(mesh_shape)
        # Initial displacement
        dx, p, _ = lpt(cosmo,
                       initial_conditions,
                       particles,
                       a=0.1,
                       order=order)
        ode_fn = ODETerm(make_diffrax_ode(cosmo, mesh_shape))
        y0 = jnp.stack([particles + dx, p])
    else:
        dx, p, _ = lpt(cosmo, initial_conditions, a=0.1, order=order)
        ode_fn = ODETerm(
            make_diffrax_ode(cosmo, mesh_shape, paint_absolute_pos=False))
        y0 = jnp.stack([dx, p])

    solver = Dopri5()
    controller = PIDController(rtol=1e-8,
                               atol=1e-8,
                               pcoeff=0.4,
                               icoeff=1,
                               dcoeff=0)

    saveat = SaveAt(t1=True)

    solutions = diffeqsolve(ode_fn,
                            solver,
                            t0=0.1,
                            t1=1.0,
                            dt0=None,
                            y0=y0,
                            stepsize_controller=controller,
                            saveat=saveat)

    if absolute_painting:
        single_device_final_field = cic_paint(jnp.zeros(shape=mesh_shape),
                                              solutions.ys[-1, 0])
    else:
        single_device_final_field = cic_paint_dx(solutions.ys[-1, 0])

    print("Done with single device run")
    # MULTI DEVICE RUN

    mesh = jax.make_mesh((1, 8), ('x', 'y'))
    sharding = NamedSharding(mesh, P('x', 'y'))
    halo_size = mesh_shape[0] // 2

    initial_conditions = lax.with_sharding_constraint(initial_conditions,
                                                      sharding)

    print(f"sharded initial conditions {initial_conditions.sharding}")

    cosmo._workspace = {}
    if absolute_painting:
        particles = uniform_particles(mesh_shape, sharding=sharding)
        # Initial displacement
        dx, p, _ = lpt(cosmo,
                       initial_conditions,
                       particles,
                       a=0.1,
                       order=order,
                       halo_size=halo_size,
                       sharding=sharding)

        ode_fn = ODETerm(
            make_diffrax_ode(cosmo,
                             mesh_shape,
                             halo_size=halo_size,
                             sharding=sharding))

        y0 = jnp.stack([particles + dx, p])
    else:
        dx, p, _ = lpt(cosmo,
                       initial_conditions,
                       a=0.1,
                       order=order,
                       halo_size=halo_size,
                       sharding=sharding)
        ode_fn = ODETerm(
            make_diffrax_ode(cosmo,
                             mesh_shape,
                             paint_absolute_pos=False,
                             halo_size=halo_size,
                             sharding=sharding))
        y0 = jnp.stack([dx, p])

    solver = Dopri5()
    controller = PIDController(rtol=1e-8,
                               atol=1e-8,
                               pcoeff=0.4,
                               icoeff=1,
                               dcoeff=0)

    saveat = SaveAt(t1=True)

    solutions = diffeqsolve(ode_fn,
                            solver,
                            t0=0.1,
                            t1=1.0,
                            dt0=None,
                            y0=y0,
                            stepsize_controller=controller,
                            saveat=saveat)

    if absolute_painting:
        multi_device_final_field = cic_paint(jnp.zeros(shape=mesh_shape),
                                             solutions.ys[-1, 0],
                                             halo_size=halo_size,
                                             sharding=sharding)
    else:
        multi_device_final_field = cic_paint_dx(solutions.ys[-1, 0],
                                                halo_size=halo_size,
                                                sharding=sharding)

    multi_device_final_field = process_allgather(multi_device_final_field,
                                                 tiled=True)

    mse = MSE(single_device_final_field, multi_device_final_field)
    print(f"MSE is  {mse}")

    assert mse < _TOLERANCE
