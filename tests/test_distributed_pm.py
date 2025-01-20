from conftest import initialize_distributed , compare_sharding

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
from jaxpm.pm import pm_forces  # noqa : E402
from jaxpm.distributed import uniform_particles , fft3d # noqa : E402
from jaxpm.painting import cic_paint, cic_paint_dx  # noqa : E402
from jaxpm.pm import lpt, make_diffrax_ode  # noqa : E402
from jaxdecomp import ShardedArray  # noqa : E402
from functools import partial  # noqa : E402
import jax_cosmo as jc  # noqa : E402
_TOLERANCE = 3.0  # 🙃🙃


@pytest.mark.distributed
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("absolute_painting", [True, False])
@pytest.mark.parametrize("shardedArrayAPI", [True, False])
def test_distrubted_pm(simulation_config, initial_conditions, cosmo, order,
                       absolute_painting,shardedArrayAPI):

    mesh_shape, box_shape = simulation_config
    # SINGLE DEVICE RUN
    cosmo._workspace = {}
    if shardedArrayAPI:
        ic = ShardedArray(initial_conditions)
    else:
        ic = initial_conditions

    if absolute_painting:
        particles = uniform_particles(mesh_shape)
        if shardedArrayAPI:
            particles = ShardedArray(particles)
        # Initial displacement
        dx, p, _ = lpt(cosmo,
                       ic,
                       particles,
                       a=0.1,
                       order=order)
        ode_fn = ODETerm(make_diffrax_ode(mesh_shape))
        y0 = jax.tree.map(lambda particles , dx , p : jnp.stack([particles + dx, p]) , particles , dx , p)
    else:
        dx, p, _ = lpt(cosmo, ic, a=0.1, order=order)
        ode_fn = ODETerm(
            make_diffrax_ode(mesh_shape, paint_absolute_pos=False))
        y0 = jax.tree.map(lambda dx , p : jnp.stack([dx, p]) , dx , p)

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
                            args=cosmo,
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

    ic = lax.with_sharding_constraint(initial_conditions,
                                                      sharding)

    print(f"sharded initial conditions {ic.sharding}")

    if shardedArrayAPI:
        ic = ShardedArray(ic , sharding)

    cosmo._workspace = {}
    if absolute_painting:
        particles = uniform_particles(mesh_shape, sharding=sharding)
        if shardedArrayAPI:
            particles = ShardedArray(particles, sharding)
        # Initial displacement
        dx, p, _ = lpt(cosmo,
                       ic,
                       particles,
                       a=0.1,
                       order=order,
                       halo_size=halo_size,
                       sharding=sharding)

        ode_fn = ODETerm(
            make_diffrax_ode(
                             mesh_shape,
                             halo_size=halo_size,
                             sharding=sharding))

        y0 = jax.tree.map(lambda particles , dx , p : jnp.stack([particles + dx, p]) , particles , dx , p)
    else:
        dx, p, _ = lpt(cosmo,
                       ic,
                       a=0.1,
                       order=order,
                       halo_size=halo_size,
                       sharding=sharding)
        ode_fn = ODETerm(
            make_diffrax_ode(
                             mesh_shape,
                             paint_absolute_pos=False,
                             halo_size=halo_size,
                             sharding=sharding))
        y0 = jax.tree.map(lambda dx , p : jnp.stack([dx, p]) , dx , p)

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
        multi_device_final_field = cic_paint(jnp.zeros(shape=mesh_shape),
                                             solutions.ys[-1, 0],
                                             halo_size=halo_size,
                                             sharding=sharding)
    else:
        multi_device_final_field = cic_paint_dx(solutions.ys[-1, 0],
                                                halo_size=halo_size,
                                                sharding=sharding)

    multi_device_final_field_g = process_allgather(multi_device_final_field,
                                                 tiled=True)

    single_device_final_field_arr, = jax.tree.leaves(single_device_final_field)
    multi_device_final_field_arr, = jax.tree.leaves(multi_device_final_field_g)
    mse = MSE(single_device_final_field_arr, multi_device_final_field_arr)
    print(f"MSE is  {mse}")

    if shardedArrayAPI:
        assert type(multi_device_final_field) == ShardedArray
        assert compare_sharding(multi_device_final_field.sharding ,  sharding)
        assert compare_sharding(multi_device_final_field.initial_sharding ,  sharding)

    assert mse < _TOLERANCE



@pytest.mark.distributed
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("absolute_painting", [True, False])
def test_distrubted_gradients(simulation_config, initial_conditions, cosmo, order,nbody_from_lpt1, nbody_from_lpt2,
                       absolute_painting):

    mesh_shape, box_shape = simulation_config
    # SINGLE DEVICE RUN
    cosmo._workspace = {}

    mesh = jax.make_mesh((1, 8), ('x', 'y'))
    sharding = NamedSharding(mesh, P('x', 'y'))
    halo_size = mesh_shape[0] // 2

    initial_conditions = lax.with_sharding_constraint(initial_conditions,
                                                      sharding)

    print(f"sharded initial conditions {initial_conditions.sharding}")


    initial_conditions = ShardedArray(initial_conditions , sharding)

    cosmo._workspace = {}

    @jax.jit
    def forward_model(initial_conditions , cosmo):


        if absolute_painting:
            particles = uniform_particles(mesh_shape, sharding=sharding)
            particles = ShardedArray(particles, sharding)
            # Initial displacement
            dx, p, _ = lpt(cosmo,
                        initial_conditions,
                        particles,
                        a=0.1,
                        order=order,
                        halo_size=halo_size,
                        sharding=sharding)

            ode_fn = ODETerm(
                make_diffrax_ode(
                                mesh_shape,
                                halo_size=halo_size,
                                sharding=sharding))

            y0 = jax.tree.map(lambda particles , dx , p : jnp.stack([particles + dx, p]) , particles , dx , p)
        else:
            dx, p, _ = lpt(cosmo,
                        initial_conditions,
                        a=0.1,
                        order=order,
                        halo_size=halo_size,
                        sharding=sharding)
            ode_fn = ODETerm(
                make_diffrax_ode(
                                mesh_shape,
                                paint_absolute_pos=False,
                                halo_size=halo_size,
                                sharding=sharding))
            y0 = jax.tree.map(lambda dx , p : jnp.stack([dx, p]) , dx , p)

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
            multi_device_final_field = cic_paint(jnp.zeros(shape=mesh_shape),
                                                solutions.ys[-1, 0],
                                                halo_size=halo_size,
                                                sharding=sharding)
        else:
            multi_device_final_field = cic_paint_dx(solutions.ys[-1, 0],
                                                    halo_size=halo_size,
                                                    sharding=sharding)

        return multi_device_final_field

    @jax.jit
    def model(initial_conditions , cosmo):

        final_field = forward_model(initial_conditions , cosmo)
        final_field, = jax.tree.leaves(final_field)
        
        return MSE(final_field,
                   nbody_from_lpt1 if order == 1 else nbody_from_lpt2)
    
    obs_val = model(initial_conditions , cosmo)

    shifted_initial_conditions = initial_conditions + jax.random.normal(jax.random.key(42) , initial_conditions.shape) * 5

    good_grads = jax.grad(model)(initial_conditions , cosmo)
    off_grads = jax.grad(model)(shifted_initial_conditions , cosmo)

    assert compare_sharding(good_grads.sharding ,  initial_conditions.sharding)
    assert compare_sharding(off_grads.sharding ,  initial_conditions.sharding)


@pytest.mark.distributed
@pytest.mark.parametrize("absolute_painting", [True, False])
def test_fwd_rev_gradients(cosmo,absolute_painting):

    mesh_shape, box_shape = (8 , 8 , 8) , (20.0 , 20.0 , 20.0)
    # SINGLE DEVICE RUN
    cosmo._workspace = {}

    mesh = jax.make_mesh((1, 8), ('x', 'y'))
    sharding = NamedSharding(mesh, P('x', 'y'))
    halo_size = mesh_shape[0] // 2

    initial_conditions = jax.random.normal(jax.random.PRNGKey(42), mesh_shape)

    initial_conditions = lax.with_sharding_constraint(initial_conditions,
                                                      sharding)

    print(f"sharded initial conditions {initial_conditions.sharding}")
    initial_conditions = ShardedArray(initial_conditions , sharding)

    cosmo._workspace = {}

    @partial(jax.jit , static_argnums=(3,4 , 5)) 
    def compute_forces(initial_conditions , cosmo , particles=None , a=0.5 , halo_size=0 , sharding=None):
        
        paint_absolute_pos = particles is not None
        if particles is None:
            particles = jax.tree.map(lambda ic : jnp.zeros_like(ic,
                                    shape=(*ic.shape, 3)) , initial_conditions)

        a = jnp.atleast_1d(a)
        E = jnp.sqrt(jc.background.Esqr(cosmo, a))
        delta_k = fft3d(initial_conditions)
        initial_force = pm_forces(particles,
                                delta=delta_k,
                                paint_absolute_pos=paint_absolute_pos,
                                halo_size=halo_size,
                                sharding=sharding)

        return initial_force[...,0]

    particles = ShardedArray(uniform_particles(mesh_shape, sharding=sharding) , sharding) if absolute_painting else None
    forces = compute_forces(initial_conditions , cosmo , particles=particles,halo_size=halo_size , sharding=sharding)
    back_gradient = jax.jacrev(compute_forces)(initial_conditions , cosmo , particles=particles,halo_size=halo_size , sharding=sharding)
    fwd_gradient = jax.jacfwd(compute_forces)(initial_conditions , cosmo , particles=particles,halo_size=halo_size , sharding=sharding)

    assert compare_sharding(forces.sharding ,  initial_conditions.sharding)
    assert compare_sharding(back_gradient[0,0,0,...].sharding ,  initial_conditions.sharding)
    assert compare_sharding(fwd_gradient.sharding ,  initial_conditions.sharding)
