from conftest import initialize_distributed

initialize_distributed()  # ignore : E402

from functools import partial  # noqa : E402

import jax  # noqa : E402
import jax.numpy as jnp  # noqa : E402
import jax_cosmo as jc  # noqa : E402
import pytest  # noqa : E402
from diffrax import SaveAt  # noqa : E402
from diffrax import Dopri5, ODETerm, PIDController, diffeqsolve
from helpers import MSE  # noqa : E402
from jax import lax  # noqa : E402
from jax.experimental.multihost_utils import process_allgather  # noqa : E402
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P  # noqa : E402
from jaxdecomp import get_fft_output_sharding

from jaxpm.distributed import uniform_particles  # noqa : E402
from jaxpm.distributed import fft3d, ifft3d
from jaxpm.painting import cic_paint, cic_paint_dx  # noqa : E402
from jaxpm.pm import lpt, make_diffrax_ode, pm_forces  # noqa : E402

_TOLERANCE = 1e-1  # 🙃🙃

pdims = [(1, 8), (8, 1), (4, 2), (2, 4)]


@pytest.mark.distributed
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("pdims", pdims)
@pytest.mark.parametrize("absolute_painting", [True, False])
def test_distrubted_pm(simulation_config, initial_conditions, cosmo, order,
                       pdims, absolute_painting):

    if absolute_painting:
        pytest.skip("Absolute painting is not recommended in distributed mode")

    painting_str = "absolute" if absolute_painting else "relative"
    print("=" * 50)
    print(f"Running with {painting_str} painting and pdims {pdims} ...")

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
        ode_fn = ODETerm(make_diffrax_ode(mesh_shape))
        y0 = jnp.stack([particles + dx, p])
    else:
        dx, p, _ = lpt(cosmo, initial_conditions, a=0.1, order=order)
        ode_fn = ODETerm(make_diffrax_ode(mesh_shape,
                                          paint_absolute_pos=False))
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
                            args=cosmo,
                            stepsize_controller=controller,
                            saveat=saveat)

    if absolute_painting:
        single_device_final_field = cic_paint(jnp.zeros(shape=mesh_shape),
                                              solutions.ys[-1, 0])
    else:
        single_device_final_field = cic_paint_dx(solutions.ys[-1, 0])

    print("Done with single device run")
    # MULTI DEVICE RUN

    mesh = jax.make_mesh(pdims, ('x', 'y'))
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
            make_diffrax_ode(mesh_shape,
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
            make_diffrax_ode(mesh_shape,
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
                            args=cosmo,
                            stepsize_controller=controller,
                            saveat=saveat)

    final_field = solutions.ys[-1, 0]
    print(f"Final field sharding is {final_field.sharding}")

    assert final_field.sharding.is_equivalent_to(sharding , ndim=3) \
        , f"Final field sharding is not correct .. should be {sharding} it is instead {final_field.sharding}"

    if absolute_painting:
        multi_device_final_field = cic_paint(jnp.zeros(shape=mesh_shape),
                                             final_field,
                                             halo_size=halo_size,
                                             sharding=sharding)
    else:
        multi_device_final_field = cic_paint_dx(final_field,
                                                halo_size=halo_size,
                                                sharding=sharding)

    multi_device_final_field = process_allgather(multi_device_final_field,
                                                 tiled=True)

    mse = MSE(single_device_final_field, multi_device_final_field)
    print(f"MSE is  {mse}")

    assert mse < _TOLERANCE


@pytest.mark.distributed
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("pdims", pdims)
def test_distrubted_gradients(simulation_config, initial_conditions, cosmo,
                              order, nbody_from_lpt1, nbody_from_lpt2, pdims):

    mesh_shape, box_shape = simulation_config
    # SINGLE DEVICE RUN
    cosmo._workspace = {}

    mesh = jax.make_mesh(pdims, ('x', 'y'))
    sharding = NamedSharding(mesh, P('x', 'y'))
    halo_size = mesh_shape[0] // 2

    initial_conditions = lax.with_sharding_constraint(initial_conditions,
                                                      sharding)

    print(f"sharded initial conditions {initial_conditions.sharding}")
    cosmo._workspace = {}

    @jax.jit
    def forward_model(initial_conditions, cosmo):

        dx, p, _ = lpt(cosmo,
                       initial_conditions,
                       a=0.1,
                       order=order,
                       halo_size=halo_size,
                       sharding=sharding)
        ode_fn = ODETerm(
            make_diffrax_ode(mesh_shape,
                             paint_absolute_pos=False,
                             halo_size=halo_size,
                             sharding=sharding))
        y0 = jax.tree.map(lambda dx, p: jnp.stack([dx, p]), dx, p)

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
                                args=cosmo,
                                stepsize_controller=controller,
                                saveat=saveat)

        multi_device_final_field = cic_paint_dx(solutions.ys[-1, 0],
                                                halo_size=halo_size,
                                                sharding=sharding)

        return multi_device_final_field

    @jax.jit
    def model(initial_conditions, cosmo):
        final_field = forward_model(initial_conditions, cosmo)
        return MSE(final_field,
                   nbody_from_lpt1 if order == 1 else nbody_from_lpt2)

    obs_val = model(initial_conditions, cosmo)

    shifted_initial_conditions = initial_conditions + jax.random.normal(
        jax.random.key(42), initial_conditions.shape) * 5

    good_grads = jax.grad(model)(initial_conditions, cosmo)
    off_grads = jax.grad(model)(shifted_initial_conditions, cosmo)

    assert good_grads.sharding.is_equivalent_to(initial_conditions.sharding,
                                                ndim=3)
    assert off_grads.sharding.is_equivalent_to(initial_conditions.sharding,
                                               ndim=3)


@pytest.mark.distributed
@pytest.mark.parametrize("pdims", pdims)
def test_fwd_rev_gradients(cosmo, pdims):

    mesh_shape, box_shape = (8, 8, 8), (20.0, 20.0, 20.0)
    cosmo._workspace = {}

    mesh = jax.make_mesh(pdims, ('x', 'y'))
    sharding = NamedSharding(mesh, P('x', 'y'))
    halo_size = mesh_shape[0] // 2

    initial_conditions = jax.random.normal(jax.random.PRNGKey(42), mesh_shape)
    initial_conditions = lax.with_sharding_constraint(initial_conditions,
                                                      sharding)
    print(f"sharded initial conditions {initial_conditions.sharding}")
    cosmo._workspace = {}

    @partial(jax.jit, static_argnums=(2, 3, 4))
    def compute_forces(initial_conditions,
                       cosmo,
                       a=0.5,
                       halo_size=0,
                       sharding=None):

        paint_absolute_pos = False
        particles = jnp.zeros_like(initial_conditions,
                                   shape=(*initial_conditions.shape, 3))

        a = jnp.atleast_1d(a)
        E = jnp.sqrt(jc.background.Esqr(cosmo, a))

        initial_conditions = jax.lax.with_sharding_constraint(
            initial_conditions, sharding)
        delta_k = fft3d(initial_conditions)
        out_sharding = get_fft_output_sharding(sharding)
        delta_k = jax.lax.with_sharding_constraint(delta_k, out_sharding)

        initial_force = pm_forces(particles,
                                  delta=delta_k,
                                  paint_absolute_pos=paint_absolute_pos,
                                  halo_size=halo_size,
                                  sharding=sharding)

        return initial_force[..., 0]

    forces = compute_forces(initial_conditions,
                            cosmo,
                            halo_size=halo_size,
                            sharding=sharding)
    back_gradient = jax.jacrev(compute_forces)(initial_conditions,
                                               cosmo,
                                               halo_size=halo_size,
                                               sharding=sharding)
    fwd_gradient = jax.jacfwd(compute_forces)(initial_conditions,
                                              cosmo,
                                              halo_size=halo_size,
                                              sharding=sharding)

    print(f"Forces sharding is {forces.sharding}")
    print(f"Backward gradient sharding is {back_gradient.sharding}")
    print(f"Forward gradient sharding is {fwd_gradient.sharding}")
    assert forces.sharding.is_equivalent_to(initial_conditions.sharding,
                                            ndim=3)
    assert back_gradient[0, 0, 0, ...].sharding.is_equivalent_to(
        initial_conditions.sharding, ndim=3)
    assert fwd_gradient.sharding.is_equivalent_to(initial_conditions.sharding,
                                                  ndim=3)


@pytest.mark.distributed
@pytest.mark.parametrize("pdims", pdims)
def test_vmap(cosmo, pdims):

    mesh_shape, box_shape = (8, 8, 8), (20.0, 20.0, 20.0)
    cosmo._workspace = {}

    mesh = jax.make_mesh(pdims, ('x', 'y'))
    sharding = NamedSharding(mesh, P('x', 'y'))
    halo_size = mesh_shape[0] // 2

    single_dev_initial_conditions = jax.random.normal(jax.random.PRNGKey(42),
                                                      mesh_shape)
    initial_conditions = lax.with_sharding_constraint(
        single_dev_initial_conditions, sharding)

    single_ics = jnp.stack([
        single_dev_initial_conditions, single_dev_initial_conditions,
        single_dev_initial_conditions
    ])
    sharded_ics = jnp.stack(
        [initial_conditions, initial_conditions, initial_conditions])
    print(f"unsharded initial conditions batch {single_ics.sharding}")
    print(f"sharded initial conditions batch {sharded_ics.sharding}")
    cosmo._workspace = {}

    @partial(jax.jit, static_argnums=(2, 3, 4))
    def compute_forces(initial_conditions,
                       cosmo,
                       a=0.5,
                       halo_size=0,
                       sharding=None):

        paint_absolute_pos = False
        particles = jnp.zeros_like(initial_conditions,
                                   shape=(*initial_conditions.shape, 3))

        a = jnp.atleast_1d(a)
        E = jnp.sqrt(jc.background.Esqr(cosmo, a))

        initial_conditions = jax.lax.with_sharding_constraint(
            initial_conditions, sharding)
        delta_k = fft3d(initial_conditions)
        out_sharding = get_fft_output_sharding(sharding)
        delta_k = jax.lax.with_sharding_constraint(delta_k, out_sharding)

        initial_force = pm_forces(particles,
                                  delta=delta_k,
                                  paint_absolute_pos=paint_absolute_pos,
                                  halo_size=halo_size,
                                  sharding=sharding)

        return initial_force[..., 0]

    def fn(ic):
        return compute_forces(ic,
                              cosmo,
                              halo_size=halo_size,
                              sharding=sharding)

    v_compute_forces = jax.vmap(fn)

    print(f"single_ics shape {single_ics.shape}")
    print(f"sharded_ics shape {sharded_ics.shape}")

    single_dev_forces = v_compute_forces(single_ics)
    sharded_forces = v_compute_forces(sharded_ics)

    assert single_dev_forces.ndim == 4
    assert sharded_forces.ndim == 4

    print(f"Sharded forces {sharded_forces.sharding}")

    assert sharded_forces[0].sharding.is_equivalent_to(
        initial_conditions.sharding, ndim=3)
    assert sharded_forces.sharding.spec[0] == None
