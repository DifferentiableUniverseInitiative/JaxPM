from conftest import initialize_distributed

initialize_distributed()  # noqa : E402

import jax  # noqa : E402
import jax.numpy as jnp  # noqa : E402
import jax_cosmo as jc  # noqa : E402
import numpy as np  # noqa : E402
import pytest  # noqa : E402
from jax import lax  # noqa : E402
from jax.experimental.multihost_utils import process_allgather  # noqa : E402
from jax.sharding import AxisType, NamedSharding  # noqa : E402
from jax.sharding import PartitionSpec as P  # noqa : E402
from numpy.testing import assert_allclose  # noqa : E402

from jaxpm.distributed import uniform_particles  # noqa : E402
from jaxpm.pm import linear_field, lpt, pm_forces  # noqa : E402

ORDERS = ['ngp', 'cic', 'tsc', 'pcs']
pdims = [(1, 8), (8, 1), (4, 2), (2, 4)]


def _evolved_displacements(cosmo,
                           mesh_shape,
                           box_shape,
                           a=0.5,
                           order=2,
                           seed=7):
    k = jnp.logspace(-3, 1, 128)
    pk = jc.power.linear_matter_power(cosmo, k)
    pk_fn = lambda x: jnp.interp(x.reshape(-1), k, pk).reshape(x.shape)
    ic = linear_field(mesh_shape,
                      box_shape,
                      pk_fn,
                      seed=jax.random.PRNGKey(seed))
    dx, _, _ = lpt(cosmo, ic, a=a, order=order)
    return dx


# ---------------------------------------------------------------------------
# Absolute (None) and displacement ('uniform') modes agree for the same config
# ---------------------------------------------------------------------------
@pytest.mark.single_device
@pytest.mark.parametrize("order", ORDERS)
def test_forces_modes_agree(simulation_config, cosmo, order):
    mesh_shape, box_shape = simulation_config
    dx = _evolved_displacements(cosmo, mesh_shape, box_shape)
    pos = uniform_particles(mesh_shape) + dx

    f_abs = pm_forces(pos,
                      mesh_shape=mesh_shape,
                      order=order,
                      initial_particles=None)
    f_dx = pm_forces(dx,
                     mesh_shape=mesh_shape,
                     order=order,
                     initial_particles='uniform')
    assert_allclose(f_abs, f_dx, rtol=1e-5, atol=1e-6)

    # the legacy paint_absolute_pos flag still bridges onto initial_particles,
    # but is deprecated and must warn
    with pytest.warns(DeprecationWarning):
        f_legacy = pm_forces(dx,
                             mesh_shape=mesh_shape,
                             order=order,
                             paint_absolute_pos=False)
    assert_allclose(f_dx, f_legacy, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# Every order produces finite, correctly-shaped, differentiable forces
# ---------------------------------------------------------------------------
@pytest.mark.single_device
@pytest.mark.parametrize("order", ORDERS)
def test_forces_orders_finite_and_grad(simulation_config, cosmo, order):
    mesh_shape, box_shape = simulation_config
    dx = _evolved_displacements(cosmo, mesh_shape, box_shape)

    forces = pm_forces(dx,
                       mesh_shape=mesh_shape,
                       order=order,
                       initial_particles='uniform')
    assert forces.shape == (*mesh_shape, 3)
    assert jnp.all(jnp.isfinite(forces))

    # reverse-mode AD must work through the scan + jax.checkpoint kernel
    grad = jax.grad(lambda d: jnp.sum(
        pm_forces(
            d, mesh_shape=mesh_shape, order=order, initial_particles='uniform')
        **2))(dx)
    assert jnp.all(jnp.isfinite(grad))


# ---------------------------------------------------------------------------
# Deconvolution is forwarded to paint and changes the forces
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_forces_deconvolution(simulation_config, cosmo):
    mesh_shape, box_shape = simulation_config
    dx = _evolved_displacements(cosmo, mesh_shape, box_shape)

    f_raw = pm_forces(dx,
                      mesh_shape=mesh_shape,
                      order='cic',
                      initial_particles='uniform',
                      deconvolution=False)
    f_dec = pm_forces(dx,
                      mesh_shape=mesh_shape,
                      order='cic',
                      initial_particles='uniform',
                      deconvolution=True)
    assert jnp.all(jnp.isfinite(f_dec))
    assert float(jnp.max(jnp.abs(f_raw - f_dec))) > 0.0


# ---------------------------------------------------------------------------
# Distributed: single-device == sharded forces (given a sufficient halo)
# ---------------------------------------------------------------------------
@pytest.mark.distributed
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("pdims", pdims)
def test_forces_distributed(simulation_config, cosmo, order, pdims):
    if jax.device_count() < 8:
        pytest.skip("requires 8 devices")
    mesh_shape, box_shape = simulation_config
    dx = _evolved_displacements(cosmo, mesh_shape, box_shape)

    single = pm_forces(dx,
                       mesh_shape=mesh_shape,
                       order=order,
                       initial_particles='uniform')

    mesh = jax.make_mesh(pdims, ('x', 'y'),
                         axis_types=(AxisType.Auto, AxisType.Auto))
    sharding = NamedSharding(mesh, P('x', 'y'))
    halo_size = (mesh_shape[0] // 2, ) * 2
    dx_sharded = lax.with_sharding_constraint(dx, sharding)
    multi = pm_forces(dx_sharded,
                      mesh_shape=mesh_shape,
                      order=order,
                      initial_particles='uniform',
                      halo_size=halo_size,
                      sharding=sharding)
    multi = process_allgather(multi, tiled=True)

    assert_allclose(single, np.asarray(multi), rtol=1e-4, atol=1e-4)
