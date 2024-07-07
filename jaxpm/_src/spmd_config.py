from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import partial
from inspect import signature
from typing import Callable , Iterable

from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding


@dataclass
class SPMDConfig():
    sharding: NamedSharding

    def __enter__(self):

        pm_operators.construct_operators(self.sharding)
        self.sharding.mesh.__enter__()
        return self.sharding.mesh

    def __exit__(self, *exc_details):
        self.sharding.mesh.__exit__(*exc_details)
        pm_operators.restore_operators(self.sharding)


@dataclass
class OpsRegistry():
    list_of_ops: list = []

    def register_operator(self, cls):
        self.list_of_ops.append(cls)
        # Register single gpu by default
        setattr(self, cls.name, cls.single_gpu_impl)

    def construct_operators(self, base_sharding=None):

        if base_sharding != None:
            for cls in self.list_of_ops:
                impl = construct_operator(cls, base_sharding)
                setattr(self, cls.name, impl)

    def restore_operators(self, base_sharding=None):

        if base_sharding != None:
            for cls in self.list_of_ops:
                setattr(self, cls.name, cls.single_gpu_impl)


pm_operators = OpsRegistry()


class CustomPartionedOperator(metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def single_gpu_impl():
        return NotImplemented

    @staticmethod
    @abstractmethod
    def multi_gpu_impl():
        return NotImplemented


class CallBackOperator(metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def single_gpu_impl():
        return NotImplemented

    @staticmethod
    @abstractmethod
    def multi_gpu_impl():
        return NotImplemented

    @staticmethod
    @abstractmethod
    def shardings_to_use_in_impl():
        return NotImplemented


class ShardedOperator(metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def single_gpu_impl():
        return NotImplemented

    @staticmethod
    @abstractmethod
    def multi_gpu_prolog():
        return NotImplemented

    @staticmethod
    @abstractmethod
    def multi_gpu_epilog():
        return NotImplemented

    @staticmethod
    @abstractmethod
    def multi_gpu_impl():
        return NotImplemented

    @staticmethod
    @abstractmethod
    def infer_sharding_from_base_sharding(base_sharding=None):
        return NotImplemented

    @staticmethod
    @abstractmethod
    def get_aux_input_from_base_sharding(base_sharding=None):
        return NotImplemented


def register_operator(cls):
    pm_operators.register_operator(cls)



def check_prolog_function(prolog_fn, impl_fn):
    prolog_sig = signature(prolog_fn)
    impl_sig = signature(impl_fn)

    if len(prolog_sig.parameters) == 0 and prolog_fn() == NotImplemented:
        return False

    prolog_return_annotation = prolog_sig.return_annotation
    if prolog_return_annotation is signature.empty:
        raise RuntimeError("Prolog function must have a return annotation")

    if isinstance(prolog_return_annotation, tuple):
        if len(prolog_return_annotation) != len(impl_sig.parameters):
            raise RuntimeError("The number of outputs of the prolog does not match the number of inputs of the impl")
    else:
        if len(impl_sig.parameters) != 1:
            raise RuntimeError("Prolog function output and impl function input count mismatch")

    return True


def check_epilog_function(epilog_fn, impl_fn):
    epilog_sig = signature(epilog_fn)
    impl_sig = signature(impl_fn)

    if len(epilog_sig.parameters) == 0 and epilog_fn() == NotImplemented:
        return False

    impl_return_annotation = impl_sig.return_annotation
    if impl_return_annotation is signature.empty:
        raise RuntimeError("Impl function must have a return annotation")

    if isinstance(impl_return_annotation, tuple):
        if len(impl_return_annotation) != len(epilog_sig.parameters):
            raise RuntimeError("The number of outputs of the impl does not match the number of inputs of the epilog")
    else:
        if len(epilog_sig.parameters) != 1:
            raise RuntimeError("Impl function output and epilog function input count mismatch")

    return True

def unpack_args(args):
    if not isinstance(args, Iterable):
        args = (args,)
    return args

def construct_operator(cls, base_sharding=None):

    if base_sharding == None:
        return
    elif not isinstance(base_sharding, NamedSharding):
        raise ValueError("base_sharding must be of type NamedSharding or None")

    if isinstance(cls, CustomPartionedOperator):
        impl = cls.multi_gpu_impl

    elif isinstance(cls, ShardedOperator):
        mesh = base_sharding.mesh
        in_spec, out_spec = cls.infer_sharding_from_base_sharding(
            base_sharding)
        __aux_input = cls.get_aux_input_from_base_sharding(base_sharding)

        if __aux_input is not None:
            multi_gpu_impl = partial(cls.multi_gpu_impl,
                                     __aux_input=__aux_input)
        else:
            multi_gpu_impl = cls.multi_gpu_impl

        multi_gpu_prolog = None
        multi_gpu_epilog = None
        if check_prolog_function(cls.multi_gpu_prolog, cls.multi_gpu_impl):
            if __aux_input is not None:
                multi_gpu_prolog = partial(cls.multi_gpu_prolog,
                                           __aux_input=__aux_input)
            else:
                multi_gpu_prolog = cls.multi_gpu_prolog
        if check_epilog_function(cls.multi_gpu_epilog, cls.multi_gpu_impl):
            if __aux_input is not None:
                multi_gpu_epilog = partial(cls.multi_gpu_epilog,
                                           __aux_input=__aux_input)
            else:
                multi_gpu_epilog = cls.multi_gpu_epilog

        sharded_impl = shard_map(multi_gpu_impl,
                                 mesh=mesh,
                                 in_spec=in_spec,
                                 out_spec=out_spec,
                                 check_rep=False)

        def impl(*params, **kwargs):
            if multi_gpu_prolog is not None:
                args = multi_gpu_prolog(*params, **kwargs)
                out = sharded_impl(*unpack_args(args))
            else:
                out = sharded_impl(*params, **kwargs)

            if multi_gpu_epilog is not None:
                out = multi_gpu_epilog(*unpack_args(out))

            return out

        return impl

    elif isinstance(cls, CallBackOperator):
        impl = partial(cls.multi_gpu_impl, base_sharding=base_sharding)

    return impl
