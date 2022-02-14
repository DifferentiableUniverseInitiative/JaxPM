from setuptools import setup, find_packages

setup(
    name='JaxPM',
    version='0.0.1',
    url='https://github.com/DifferentiableUniverseInitiative/JaxPM',
    author='JaxPM developers',
    description='A dead simple FastPM implementation in JAX',
    packages=find_packages(),    
    install_requires=['jax', 'jax_cosmo'],
)