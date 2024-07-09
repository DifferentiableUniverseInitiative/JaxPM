from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import partial
from inspect import signature

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jaxdecomp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxdecomp import halo_exchange
from jaxdecomp.fft import pfft3d, pifft3d

from jaxpm._src.spmd_config import (CallBackOperator, CustomPartionedOperator,
                                    ShardedOperator, register_operator)


class FFTOperator(CustomPartionedOperator):

    name = 'fftn'

    def single_gpu_impl(x):
        return jnp.fft.fftn(x).transpose([1, 2, 0])

    def multi_gpu_impl(x):
        return pfft3d(x)


class IFFTOperator(CustomPartionedOperator):

    name = 'ifftn'

    def single_gpu_impl(x):
        return jnp.fft.ifftn(x).transpose([2, 0, 1])

    def multi_gpu_impl(x):
        return pifft3d(x)


class HaloExchangeOperator(CustomPartionedOperator):

    name = 'halo_exchange'

    # Halo exchange does nothing on a single GPU
    # Inside a jit , this will be optimized out
    def single_gpu_impl(x):
        return x

    def multi_gpu_impl(x):
        return halo_exchange(x)


# Padding and unpadding operators should not do anything in case of single GPU
# Since there is no halo exchange for a single GPU
# Inside a jit , this will be optimized out


class PaddingOperator(ShardedOperator):

    name = 'slice_pad'

    def single_gpu_impl(x, pad_width):
        return x

    def multi_gpu_impl(x, pad_width):
        return jnp.pad(x, pad_width)

    def infer_sharding_from_base_sharding(base_sharding):

        in_spec = base_sharding, P()
        out_spec = base_sharding

        return in_spec, out_spec


class UnpaddingOperator(ShardedOperator):

    name = 'slice_unpad'

    def single_gpu_impl(x, pad_width):
        return x

    def multi_gpu_impl(x, pad_width):

        # WARNING : unequal halo size is not supported
        halo_x, _ = pad_width[0]
        halo_y, _ = pad_width[0]

        # Apply corrections along x
        x = x.at[halo_x:halo_x + halo_x // 2].add(x[:halo_x // 2])
        x = x.at[-(halo_x + halo_x // 2):-halo_x].add(x[-halo_x // 2:])
        # Apply corrections along y
        x = x.at[:, halo_y:halo_y + halo_y // 2].add(x[:, :halo_y // 2])
        x = x.at[:, -(halo_y + halo_y // 2):-halo_y].add(x[:, -halo_y // 2:])

        return x[halo_x:-halo_x, halo_y:-halo_y, :]

    def infer_sharding_from_base_sharding(base_sharding):

        in_spec = base_sharding, P()
        out_spec = base_sharding

        return in_spec, out_spec


class NormalFieldOperator(CallBackOperator):

    name = 'normal'

    def single_gpu_impl(shape, key, dtype='float32'):
        return jax.random.normal(key, shape, dtype=dtype)

    def multi_gpu_impl(shape, key, dtype='float32', base_sharding=None):

        assert (isinstance(base_sharding, NamedSharding))
        sharding = NormalFieldOperator.shardings_to_use_in_impl(base_sharding)

        def get_axis_size(sharding, index):
            axis_name = sharding.spec[index]
            if axis_name == None:
                return 1
            else:
                return sharding.mesh.shape[sharding.spec[index]]

        pdims = [get_axis_size(sharding, i) for i in range(2)]
        local_mesh_shape = [
            shape[0] // pdims[1], shape[1] // pdims[0], shape[2]
        ]

        return jax.make_array_from_single_device_arrays(
            shape=shape,
            sharding=sharding,
            arrays=[jax.random.normal(key, local_mesh_shape, dtype=dtype)])

    def shardings_to_use_in_impl(base_sharding):
        return base_sharding


class FFTKOperator(CallBackOperator):
    name = 'fftk'

    def single_gpu_impl(shape, symmetric=True, finite=False, dtype=np.float32):
        k = []
        for d in range(len(shape)):
            kd = np.fft.fftfreq(shape[d])
            kd *= 2 * np.pi
            kdshape = np.ones(len(shape), dtype='int')
            if symmetric and d == len(shape) - 1:
                kd = kd[:shape[d] // 2 + 1]
            kdshape[d] = len(kd)
            kd = kd.reshape(kdshape)

            k.append(kd.astype(dtype))
        del kd, kdshape
        return k

    def multi_gpu_impl(shape,
                       symmetric=True,
                       finite=False,
                       dtype=np.float32,
                       base_sharding=None):

        assert (isinstance(base_sharding, NamedSharding))
        kvec = FFTKOperator.single_gpu_impl(shape, symmetric, finite, dtype)

        z_sharding, y_sharding = FFTKOperator.shardings_to_use_in_impl(shape)

        return [
            jax.make_array_from_callback(
                (shape[0], 1, 1),
                sharding=z_sharding,
                data_callback=lambda x: kvec[0].reshape([-1, 1, 1])[x]),
            jax.make_array_from_callback(
                (1, shape[1], 1),
                sharding=y_sharding,
                data_callback=lambda x: kvec[1].reshape([1, -1, 1])[x]),
            kvec[2].reshape([1, 1, -1])
        ]

    @staticmethod
    def shardings_to_use_in_impl(base_sharding):
        spec = base_sharding.spec

        z_sharding = NamedSharding(P(spec[0], None, None))
        y_sharding = NamedSharding(P(None, spec[1], None))

        return z_sharding, y_sharding


class GenerateParticlesOperator(CallBackOperator):

    name = 'generate_initial_positions'

    def single_gpu_impl(shape):
        return jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in shape]),
                         axis=-1)

    def multi_gpu_impl(shape, base_sharding=None):
        assert (isinstance(base_sharding, NamedSharding))
        sharding = GenerateParticlesOperator.shardings_to_use_in_impl(
            base_sharding)

        return jax.make_array_from_callback(
            shape=tuple([*shape, 3]),
            sharding=sharding,
            data_callback=lambda x: jnp.stack(jnp.meshgrid(
                jnp.arange(shape[0])[x[0]],
                jnp.arange(shape[1])[x[1]],
                jnp.arange(shape[2]),
                indexing='ij'),
                                              axis=-1))

    def shardings_to_use_in_impl(base_sharding):
        return base_sharding


class InterpolateICOperator(ShardedOperator):

    name = 'interpolate_ic'

    # TODO : find a way to allow using different transfer fn
    def single_gpu_impl(kfield, kk, cosmo: jc.Cosmology, box_size):

        k = jnp.logspace(-4, 2, 128)  # I don't understand why 256?

        mesh_shape = kfield.shape
        pk = jc.power.linear_matter_power(cosmo, k)
        pk = pk * (mesh_shape[0] / box_size[0]) * (
            mesh_shape[1] / box_size[1]) * (mesh_shape[2] / box_size[2])
        print(f"kk {kk.shape}")
        print(f"kk.flatten() {kk.flatten().shape}")
        delta_k = kfield * jc.scipy.interpolate.interp(
            kk.flatten(), k, pk**0.5).reshape(kfield.shape)

        return delta_k

    def multi_gpu_impl(kfield,
                       kk,
                       cosmo: jc.Cosmology,
                       box_size,
                       k=jnp.logspace(-4, 2, 256)):

        mesh_shape = kfield.shape
        pk = jc.power.linear_matter_power(cosmo, k)
        pk = pk * (mesh_shape[0] / box_size[0]) * (
            mesh_shape[1] / box_size[1]) * (mesh_shape[2] / box_size[2])
        delta_k = kfield * jc.scipy.interpolate.interp(
            kk.flatten(), k, pk**0.5).reshape(kfield.shape)

        return delta_k

    def infer_sharding_from_base_sharding(base_sharding):

        in_spec = base_sharding, base_sharding, P(), P(), P()
        out_spec = base_sharding

        return in_spec, out_spec


register_operator(FFTOperator)
register_operator(IFFTOperator)
register_operator(HaloExchangeOperator)
register_operator(PaddingOperator)
register_operator(UnpaddingOperator)
register_operator(NormalFieldOperator)
register_operator(FFTKOperator)
register_operator(GenerateParticlesOperator)
register_operator(InterpolateICOperator)
