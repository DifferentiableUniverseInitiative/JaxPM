from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Sequence

import jax
import jax_cosmo as jc
from diffrax import (AbstractSolver, AbstractStepSizeController,
                     ConstantStepSize, ODETerm, SaveAt, Tsit5, diffeqsolve)
from jax import numpy as jnp
from jax_cosmo import Cosmology
from jaxtyping import Array, Bool, PyTree, Real, Shaped

import jaxpm as jpm
from jaxpm.growth import (growth_factor, growth_factor_second, growth_rate,
                          growth_rate_second)


class PMSolverStatus(Enum):
    INIT = 0
    LPT = 1
    LPT2 = 2
    ODE = 3
    DONE = 4


@partial(jax.tree_util.register_dataclass,
         data_fields=['displacements', 'velocities', 'cosmo'],
         meta_fields=['solver_stats', 'status', 'kvec'])
@dataclass
class State(object):
    displacements: Array
    velocities: Array
    cosmo: Cosmology
    kvec: Sequence
    solver_stats: dict[str, Any] = None
    status: PMSolverStatus = PMSolverStatus.INIT


class FastPM(object):
    initial_delta_k: Array
    kvec: list[Array, Array, Array]

    def init_state(self, cosmo, particules, kvec, initial_field, box_size):
        self.initial_delta_k = jpm.ops.fftn(initial_field)
        self.box_size = box_size
        # Check sharding on this
        zeros = jnp.zeros_like(particules)
        state = State(displacements=zeros,
                      velocities=zeros,
                      cosmo=cosmo,
                      kvec=kvec)

        return state

    def compute_initial_forces(self, state, delta_k):

        #TODO this must done in a function generat_ic
        mesh_shape = state.displacements.shape[:3]
        box_size = self.box_size
        ky, kz, kx = state.kvec
        kk = jnp.sqrt((kx / box_size[0] * mesh_shape[0])**2 +
                      (ky / box_size[1] * mesh_shape[1])**2 +
                      (kz / box_size[1] * mesh_shape[1])**2)
        delta_k = jpm.ops.interpolate_ic(delta_k, kk, state.cosmo, box_size)
        kernel_lap = jnp.where(
            kk == 0, 1.,
            1. / (kx**2 + ky**2 + kz**2))  # Laplace kernel + longrange
        pot_k = delta_k * kernel_lap
        # Forces have to be a Z pencil because they are going to be IFFT back to X pencil
        forces_k = jnp.stack([
            pot_k * 1j / 6.0 *
            (8 * jnp.sin(kx) - jnp.sin(2 * kx)), pot_k * 1j / 6.0 *
            (8 * jnp.sin(ky) - jnp.sin(2 * ky)), pot_k * 1j / 6.0 *
            (8 * jnp.sin(kz) - jnp.sin(2 * kz))
        ],
                             axis=-1)

        init_force = jnp.stack(
            [jpm.ops.ifftn(forces_k[..., i]).real for i in range(3)], axis=-1)

        return init_force, delta_k

    def compute_ode_forces(self, state):

        mesh_shape = self.initial_delta_k.shape
        box_size = self.box_size
        print(f"type of state {type(state)}")

        pos = jnp.array(state.displacements)

        ky, kz, kx = state.kvec
        kk = jnp.sqrt((kx / box_size[0] * mesh_shape[0])**2 +
                      (ky / box_size[1] * mesh_shape[1])**2 +
                      (kz / box_size[1] * mesh_shape[1])**2)
        delta_k = jpm.painting.cic_paint_dx(pos)
        kernel_lap = jnp.where(
            kk == 0, 1.,
            1. / (kx**2 + ky**2 + kz**2))  # Laplace kernel + longrange
        pot_k = delta_k * kernel_lap
        # Forces have to be a Z pencil because they are going to be IFFT back to X pencil
        forces_k = jnp.stack([
            pot_k * 1j / 6.0 *
            (8 * jnp.sin(kx) - jnp.sin(2 * kx)), pot_k * 1j / 6.0 *
            (8 * jnp.sin(ky) - jnp.sin(2 * ky)), pot_k * 1j / 6.0 *
            (8 * jnp.sin(kz) - jnp.sin(2 * kz))
        ],
                             axis=-1)

        forces = jnp.stack([
            jpm.painting.cic_read_dx(jpm.ops.ifftn(forces_k[..., i])).real
            for i in range(3)
        ],
                           axis=-1)
        forces = forces * 1.5 * state.cosmo.Omega_m

        return forces

    def make_ode_fn(self):

        def ode_fn(a, state, args):

            print(f"helo")
            forces = self.compute_ode_forces(state)

            # Computes the update of position (drift)
            dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(
                state.cosmo, a))) * state.velocities

            # Computes the update of velocity (kick)
            dvel = 1. / (a**2 *
                         jnp.sqrt(jc.background.Esqr(state.cosmo, a))) * forces

            state = State(displacements=dpos,
                          velocities=dvel,
                          cosmo=state.cosmo,
                          kvec=state.kvec,
                          status=PMSolverStatus.ODE)
            return state

        return ode_fn

    def compute_lpt2_source(self, delta_k):

        mesh_shape = self.initial_delta_k.shape
        box_size = self.box_size

        ky, kz, kx = self.kvec
        kk = jnp.sqrt((kx / box_size[0] * mesh_shape[0])**2 +
                      (ky / box_size[1] * mesh_shape[1])**2 +
                      (kz / box_size[1] * mesh_shape[1])**2)
        invlaplace_kernel = -jnp.where(kk == 0, 1., 1. /
                                       (kx**2 + ky**2 + kz**2))

        pot_k = delta_k * invlaplace_kernel

        # Taken from https://github.com/hsimonfroy/montecosmo
        # Based on https://arxiv.org/abs/0910.0258
        delta2 = 0
        shear_acc = 0
        for i, ki in enumerate(self.kvec):
            # Add products of diagonal terms = 0 + s11*s00 + s22*(s11+s00)...
            shear_ii = jpm.ops.ifft(-ki**2 * pot_k)
            delta2 += shear_ii * shear_acc
            shear_acc += shear_ii

            for kj in self.kvec[i + 1:]:
                # Substract squared strict-up-triangle terms
                delta2 -= jpm.ops.ifft(-ki * kj * pot_k)**2

        return delta2

    def lpt(self, state, a=0.1):

        a = jnp.atleast_1d(a)

        if state.status != PMSolverStatus.INIT:
            raise ValueError(
                f"LPT simulation has to be done before the other steps")

        init_force, _ = self.compute_initial_forces(state,
                                                    self.initial_delta_k)

        dx = growth_factor(state.cosmo, a) * init_force

        p = a**2 * growth_rate(state.cosmo, a) * jnp.sqrt(
            jc.background.Esqr(state.cosmo, a)) * dx

        return State(displacements=dx,
                     velocities=p,
                     cosmo=state.cosmo,
                     kvec=state.kvec,
                     status=PMSolverStatus.LPT)

    def lpt2(self, state, a=0.1):
        a = jnp.atleast_1d(a)

        if state.status != PMSolverStatus.INIT:
            raise ValueError(
                f"LPT2 simulation has to be done in the beginning")

        init_force, delta_k = self.compute_initial_forces(
            state, self.initial_delta_k)

        dx = growth_factor(state.cosmo, a) * init_force

        p = a**2 * growth_rate(state.cosmo, a) * jnp.sqrt(
            jc.background.Esqr(state.cosmo, a)) * dx

        delta2 = self.compute_lpt2_source(delta_k)

        init_force2 = self.compute_initial_forces(state, delta2)

        dx2 = 3 / 7 * growth_factor_second(state.cosmo, a) * init_force2
        p2 = a**2 * growth_rate_second(state.cosmo, a) * jnp.sqrt(
            jc.background.Esqr(state.cosmo, a)) * dx2

        dx += dx2
        p += p2

        return State(displacements=dx,
                     velocities=p,
                     cosmo=state.cosmo,
                     status=PMSolverStatus.LPT2)

    def nbody(self,
              state,
              solver: AbstractSolver,
              stepsize_controller: AbstractStepSizeController,
              t0=0.1,
              t1=1,
              dt0=0.01):

        if state.status == PMSolverStatus.INIT:
            state = self.lpt(state, a=t0)
        elif state.status == PMSolverStatus.ODE or \
            state.status == PMSolverStatus.DONE:
            raise ValueError(f"nbody already done on state {state.status}")

        ode_fn = self.make_ode_fn()

        state.status = PMSolverStatus.ODE
        solution = diffeqsolve(ODETerm(ode_fn),
                               solver,
                               t0=t0,
                               t1=t1,
                               dt0=dt0,
                               y0=state,
                               saveat=SaveAt(t1=True),
                               args=None,
                               stepsize_controller=stepsize_controller)

        return State(displacements=solution.ys.displacements[-1],
                     velocities=solution.ys.velocities[-1],
                     cosmo=state.cosmo,
                     kvec=state.kvec,
                     status=PMSolverStatus.DONE,
                     solver_stats=solution.stats)
