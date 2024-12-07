import jax.numpy as jnp

def MSE(x , y):
    return jnp.mean((x - y)**2)

def MSE_3D(x , y):
    return ((x - y)**2).mean(axis=0)

def MSRE(x , y):
    return jnp.mean(((x - y)/ y)**2)