# Parameterized fixture for mesh_shape
import os

import pytest

os.environ["EQX_ON_ERROR"] = "nan"
setup_done = False
on_cluster = False


def is_on_cluster():
    global on_cluster
    return on_cluster


def initialize_distributed():
    global setup_done
    global on_cluster
    if not setup_done:
        if "SLURM_JOB_ID" in os.environ:
            on_cluster = True
            print("Running on cluster")
            import jax
            jax.distributed.initialize()
            setup_done = True
            on_cluster = True
        else:
            print("Running locally")
            setup_done = True
            on_cluster = False
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
            os.environ[
                "XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
            import jax


@pytest.fixture(
    scope="session",
    params=[
        ((32, 32, 32), (256., 256., 256.)),  # BOX
        ((32, 32, 64), (256., 256., 512.)),  # RECTANGULAR
    ])
def simulation_config(request):
    return request.param


@pytest.fixture(scope="session", params=[0.1, 0.5, 0.8])
def lpt_scale_factor(request):
    return request.param


@pytest.fixture(scope="session")
def cosmo():
    from functools import partial

    from jax_cosmo import Cosmology
    Planck18 = partial(
        Cosmology,
        # Omega_m = 0.3111
        Omega_c=0.2607,
        Omega_b=0.0490,
        Omega_k=0.0,
        h=0.6766,
        n_s=0.9665,
        sigma8=0.8102,
        w0=-1.0,
        wa=0.0,
    )

    return Planck18()


@pytest.fixture(scope="session")
def particle_mesh(simulation_config):
    from pmesh.pm import ParticleMesh
    mesh_shape, box_shape = simulation_config
    return ParticleMesh(BoxSize=box_shape, Nmesh=mesh_shape, dtype='f4')


@pytest.fixture(scope="session")
def fpm_initial_conditions(cosmo, particle_mesh):
    import jax_cosmo as jc
    import numpy as np
    from jax import numpy as jnp

    # Generate initial particle positions
    grid = particle_mesh.generate_uniform_particle_grid(shift=0).astype(
        np.float32)
    # Interpolate with linear_matter spectrum to get initial density field
    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(cosmo, k)

    def pk_fn(x):
        return jnp.interp(x.reshape([-1]), k, pk).reshape(x.shape)

    whitec = particle_mesh.generate_whitenoise(42,
                                               type='complex',
                                               unitary=False)
    lineark = whitec.apply(lambda k, v: jnp.sqrt(
        pk_fn(jnp.sqrt(sum(ki**2 for ki in k)))) * v * jnp.sqrt(
            (1 / v.BoxSize).prod()))
    init_mesh = lineark.c2r().value  # XXX

    return lineark, grid, init_mesh


@pytest.fixture(scope="session")
def initial_conditions(fpm_initial_conditions):
    _, _, init_mesh = fpm_initial_conditions
    return init_mesh


@pytest.fixture(scope="session")
def solver(cosmo, particle_mesh):
    from fastpm.core import Cosmology as FastPMCosmology
    from fastpm.core import Solver
    ref_cosmo = FastPMCosmology(cosmo)
    return Solver(particle_mesh, ref_cosmo, B=1)


@pytest.fixture(scope="session")
def fpm_lpt1(solver, fpm_initial_conditions, lpt_scale_factor):

    lineark, grid, _ = fpm_initial_conditions
    statelpt = solver.lpt(lineark, grid, lpt_scale_factor, order=1)
    return statelpt


@pytest.fixture(scope="session")
def fpm_lpt1_field(fpm_lpt1, particle_mesh):
    return particle_mesh.paint(fpm_lpt1.X).value


@pytest.fixture(scope="session")
def fpm_lpt2(solver, fpm_initial_conditions, lpt_scale_factor):

    lineark, grid, _ = fpm_initial_conditions
    statelpt = solver.lpt(lineark, grid, lpt_scale_factor, order=2)
    return statelpt


@pytest.fixture(scope="session")
def fpm_lpt2_field(fpm_lpt2, particle_mesh):
    return particle_mesh.paint(fpm_lpt2.X).value


@pytest.fixture(scope="session")
def nbody_from_lpt1(solver, fpm_lpt1, particle_mesh, lpt_scale_factor):
    import numpy as np
    from fastpm.core import leapfrog

    if lpt_scale_factor == 0.8:
        pytest.skip("Do not run nbody simulation from scale factor 0.8")

    stages = np.linspace(lpt_scale_factor, 1.0, 10, endpoint=True)

    finalstate = solver.nbody(fpm_lpt1, leapfrog(stages))
    fpm_mesh = particle_mesh.paint(finalstate.X).value

    return fpm_mesh


@pytest.fixture(scope="session")
def nbody_from_lpt2(solver, fpm_lpt2, particle_mesh, lpt_scale_factor):
    import numpy as np
    from fastpm.core import leapfrog

    if lpt_scale_factor == 0.8:
        pytest.skip("Do not run nbody simulation from scale factor 0.8")

    stages = np.linspace(lpt_scale_factor, 1.0, 10, endpoint=True)

    finalstate = solver.nbody(fpm_lpt2, leapfrog(stages))
    fpm_mesh = particle_mesh.paint(finalstate.X).value

    return fpm_mesh
