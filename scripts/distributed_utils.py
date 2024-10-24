import os
from math import prod

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
                "XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
            import jax


def compare_sharding(sharding1, sharding2):
    from jaxdecomp._src.spmd_ops import get_pdims_from_sharding
    pdims1 = get_pdims_from_sharding(sharding1)
    pdims2 = get_pdims_from_sharding(sharding2)
    pdims1 = pdims1 + (1, ) * (3 - len(pdims1))
    pdims2 = pdims2 + (1, ) * (3 - len(pdims2))
    return pdims1 == pdims2


def replace_none_or_zero(value):
    # Replace None or 0 with 1
    return 0 if value is None else value


def process_slices(slices_tuple):

    start_product = 1
    stop_product = 1

    for s in slices_tuple:
        # Multiply the start and stop values, replacing None/0 with 1
        start_product *= replace_none_or_zero(s.start)
        stop_product *= replace_none_or_zero(s.stop)

    # Return the sum of the two products
    return int(start_product + stop_product)


def device_arange(pdims):
    import jax
    from jax import numpy as jnp
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    devices = mesh_utils.create_device_mesh(pdims)
    mesh = Mesh(devices.T, axis_names=('z', 'y'))
    sharding = NamedSharding(mesh, P('z', 'y'))

    def generate_aranged(x):
        x_start = replace_none_or_zero(x[0].start)
        y_start = replace_none_or_zero(x[1].start)
        a = jnp.array([[x_start + y_start * pdims[0]]])
        print(f"index is {x} and value is {a}")
        return a

    aranged = jax.make_array_from_callback(mesh.devices.shape,
                                           sharding,
                                           data_callback=generate_aranged)

    return aranged


def create_ones_spmd_array(global_shape, pdims):

    import jax
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    size = jax.device_count()
    assert (len(global_shape) == 3)
    assert (len(pdims) == 2)
    assert (
        prod(pdims) == size
    ), "The product of pdims must be equal to the number of MPI processes"

    local_shape = (global_shape[0] // pdims[1], global_shape[1] // pdims[0],
                   global_shape[2])

    # Remap to the global array from the local slice
    devices = mesh_utils.create_device_mesh(pdims)
    mesh = Mesh(devices.T, axis_names=('z', 'y'))
    sharding = NamedSharding(mesh, P('z', 'y'))
    global_array = jax.make_array_from_callback(
        global_shape,
        sharding,
        data_callback=lambda _: jax.numpy.ones(local_shape))

    return global_array, mesh


# Helper function to create a 3D array and remap it to the global array
def create_spmd_array(global_shape, pdims):

    import jax
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    size = jax.device_count()
    assert (len(global_shape) == 3)
    assert (len(pdims) == 2)
    assert (
        prod(pdims) == size
    ), "The product of pdims must be equal to the number of MPI processes"

    local_shape = (global_shape[0] // pdims[1], global_shape[1] // pdims[0],
                   global_shape[2])

    # Remap to the global array from the local slicei
    devices = mesh_utils.create_device_mesh(pdims)
    mesh = Mesh(devices.T, axis_names=('z', 'y'))
    sharding = NamedSharding(mesh, P('z', 'y'))
    global_array = jax.make_array_from_callback(
        global_shape,
        sharding,
        data_callback=lambda x: jax.random.normal(
            jax.random.PRNGKey(process_slices(x)), local_shape))

    return global_array, mesh
