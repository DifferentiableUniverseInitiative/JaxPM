from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from jax_cosmo import Cosmology
from jaxtyping import Array, Bool, PyTree, Real, Shaped


class PMSolverStatus(Enum):
    INIT = 0
    LPT = 1
    LPT2 = 2
    ODE = 3


@register_pytree_node_class
@dataclass
class State(object):
    displacements: Tuple[int, int, int, 3]
    velocities: Tuple[int, int, int, 3]
    cosmo: Cosmology
    kvec: list[Array, Array, Array]
    solver_stats: dict[str, Any]
    status: PMSolverStatus

    def tree_flatten(self):
        children = (self.displacements, self.velocities, self.cosmo)
        aux_data = (self.kvec, self.solver_stats, self.status)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        return cls(*children)


class FastPM(object):
    initial_conditions: Array

    def init_state(self, cosmo, particules, kvec, initial_conditions):
        self.initial_conditions = initial_conditions
        # Check sharding on this
        zeros = jnp.zeros_like(particules)
        state = State(displacements=zeros,
                      velocities=zeros,
                      cosmo=cosmo,
                      kvec=kvec)

        return state

    def lpt(state, a=0.1):
        pass

    def lpt2(state, a=0.1):
        pass

    def nbody(state, solver, stepsize_controller, t0=0.1, t1=1, dt0=0.01):
        pass
