import jax.numpy as jnp


def MSE(x, y):
    return ((x - y)**2).mean()


def MSE_3D(x, y):
    return ((x - y)**2).mean(axis=0)


def MSRE(x, y):
    return (((x - y) / y)**2).mean()
