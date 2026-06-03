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
from jaxpm.kernels import gridding_shotnoise_kernel  # noqa : E402
from jaxpm.painting import cic_read  # noqa : E402
from jaxpm.painting import (cic_paint, cic_paint_dx, cic_read_dx,
                            compensate_cic, paint, readout)
from jaxpm.pm import linear_field, lpt  # noqa : E402
from jaxpm.utils import power_spectrum  # noqa : E402

ORDERS = ['ngp', 'cic', 'tsc', 'pcs']
pdims = [(1, 8), (8, 1), (4, 2), (2, 4)]


def _evolved_displacements(cosmo,
                           mesh_shape,
                           box_shape,
                           a=0.5,
                           order=2,
                           seed=5):
    """Displacements from a 2LPT-evolved field -- clustered enough that the
    aliased shot noise is actually present (a lattice IC suppresses it)."""
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
# Equivalence with the deprecated CIC implementation
# ---------------------------------------------------------------------------
@pytest.mark.single_device
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_new_cic_matches_old(simulation_config, cosmo):
    mesh_shape, box_shape = simulation_config
    dx = _evolved_displacements(cosmo, mesh_shape, box_shape)
    # Stress the divergence surface: cell-0, cell-(N-1), negative displacements.
    dx = dx.at[0, 0, 0].set(jnp.array([-2.3, -5.1, mesh_shape[2] + 0.7]))
    dx = dx.at[-1, -1, -1].set(jnp.array([3.4, -0.5, -8.2]))
    pos = (uniform_particles(mesh_shape) + dx).reshape(-1, 3)
    field = jax.random.normal(jax.random.PRNGKey(1), mesh_shape)

    # paint -- displacement and absolute modes
    assert_allclose(paint(dx, initial_particles='uniform', order='cic'),
                    cic_paint_dx(dx),
                    rtol=1e-4,
                    atol=1e-4)
    assert_allclose(paint(pos.reshape(*mesh_shape, 3),
                          grid_mesh=jnp.zeros(mesh_shape),
                          order='cic'),
                    cic_paint(jnp.zeros(mesh_shape), pos),
                    rtol=1e-4,
                    atol=1e-4)

    # readout -- displacement and absolute modes
    assert_allclose(readout(field,
                            dx,
                            initial_particles='uniform',
                            order='cic'),
                    cic_read_dx(field, dx),
                    rtol=1e-4,
                    atol=1e-4)
    assert_allclose(readout(field, pos.reshape(*mesh_shape, 3),
                            order='cic').reshape(-1),
                    cic_read(field, pos),
                    rtol=1e-4,
                    atol=1e-4)


# ---------------------------------------------------------------------------
# Partition of unity: every order conserves total mass
# ---------------------------------------------------------------------------
@pytest.mark.single_device
@pytest.mark.parametrize("order", ORDERS)
def test_partition_of_unity(simulation_config, cosmo, order):
    mesh_shape, box_shape = simulation_config
    dx = _evolved_displacements(cosmo, mesh_shape, box_shape)
    n_part = np.prod(mesh_shape)
    for mode, arr, kw in [('dx', dx, dict(initial_particles='uniform')),
                          ('abs', (uniform_particles(mesh_shape) + dx),
                           dict(grid_mesh=jnp.zeros(mesh_shape)))]:
        field = paint(arr, order=order, **kw)
        assert_allclose(float(field.sum()), n_part, rtol=1e-3), mode


# ---------------------------------------------------------------------------
# The real correctness gate: after deconvolution + dealiasing, all four orders
# recover the *same* power spectrum (validates the per-order windows AND the
# shot-noise polynomials together).
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_orders_recover_same_pk(cosmo):
    mesh_shape, box_shape = (64, 64, 64), (256., 256., 256.)
    dx = _evolved_displacements(cosmo, mesh_shape, box_shape, a=1.0, order=2)
    nbar = np.prod(mesh_shape) / np.prod(box_shape)  # one particle per cell
    knyq = np.pi * min(np.array(mesh_shape) / np.array(box_shape))

    corrected = {}
    ks = None
    for o in ORDERS:
        field = paint(dx, initial_particles='uniform', order=o)
        ks, pk = power_spectrum(field,
                                box_shape=box_shape,
                                compensate_order=o,
                                shotnoise=(o, nbar))
        corrected[o] = np.asarray(pk)
    ks = np.asarray(ks)
    sel = (ks > 0) & (ks < 0.5 * knyq)
    assert sel.sum() >= 4

    cic = corrected['cic']
    # CIC/TSC/PCS must agree tightly; NGP (the crudest scheme) is looser.
    for o, tol in [('tsc', 0.05), ('pcs', 0.05), ('ngp', 0.30)]:
        ratio = corrected[o][sel] / cic[sel]
        assert np.all(np.abs(ratio - 1) < tol), (o, ratio)


# ---------------------------------------------------------------------------
# Deconvolution and standalone kernels
# ---------------------------------------------------------------------------
@pytest.mark.single_device
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_deconvolution_matches_compensate_cic(simulation_config, cosmo):
    mesh_shape, box_shape = simulation_config
    dx = _evolved_displacements(cosmo, mesh_shape, box_shape)
    new = paint(dx,
                initial_particles='uniform',
                order='cic',
                deconvolution=True)
    ref = compensate_cic(cic_paint_dx(dx))
    assert_allclose(new, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.single_device
def test_shotnoise_kernel_formulas():
    k = jnp.linspace(-np.pi, np.pi, 33)
    zeros = jnp.zeros_like(k)
    s2 = np.sin(np.asarray(k) / 2.0)**2
    expected = {
        'ngp': 1.0 + 0 * s2,
        'cic': 1 - 2 / 3 * s2,
        'tsc': 1 - s2 + 2 / 15 * s2**2,
        'pcs': 1 - 4 / 3 * s2 + 2 / 5 * s2**2 - 4 / 315 * s2**3,
    }
    for o in ORDERS:
        # only kx varies, ky=kz=0 -> per-dim(0)=1, so the product equals per-dim(kx)
        got = gridding_shotnoise_kernel([k, zeros, zeros], o)
        assert_allclose(np.asarray(got), expected[o], rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Distributed: single-device == sharded (given a sufficient halo)
# ---------------------------------------------------------------------------
@pytest.mark.distributed
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("pdims", pdims)
def test_paint_distributed(simulation_config, cosmo, order, pdims):
    if jax.device_count() < 8:
        pytest.skip("requires 8 devices")
    mesh_shape, box_shape = simulation_config
    dx = _evolved_displacements(cosmo, mesh_shape, box_shape)

    single = paint(dx, initial_particles='uniform', order=order)

    mesh = jax.make_mesh(pdims, ('x', 'y'),
                         axis_types=(AxisType.Auto, AxisType.Auto))
    sharding = NamedSharding(mesh, P('x', 'y'))
    halo_size = (mesh_shape[0] // 2, ) * 2  # generous: fits NGP..PCS stencils
    dx_sharded = lax.with_sharding_constraint(dx, sharding)
    multi = paint(dx_sharded,
                  initial_particles='uniform',
                  order=order,
                  halo_size=halo_size,
                  sharding=sharding)
    multi = process_allgather(multi, tiled=True)

    assert_allclose(single, np.asarray(multi), rtol=1e-4, atol=1e-4)
