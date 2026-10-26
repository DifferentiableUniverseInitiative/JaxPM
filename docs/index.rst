JaxPM documentation
===================

JaxPM is a JAX-powered cosmological Particle-Mesh (PM) N-body solver: a differentiable,
distributable PM implementation that scales from a single GPU to multi-GPU / multi-host
systems.

The tutorials below are executable Jupyter notebooks, rendered here from their committed
outputs (the documentation build does not run them). Open any notebook in Colab via the
badge at its top to run it yourself.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   notebooks/01-Introduction
   notebooks/02-Advanced_usage
   notebooks/02b-Painting_and_Deconvolution

.. toctree::
   :maxdepth: 1
   :caption: Multi-GPU / multi-host

   notebooks/03-MultiGPU_PM_Halo
   notebooks/04-MultiGPU_PM_Solvers
   notebooks/05-MultiHost_PM
   notebooks/06-Animating_PM_Fields

.. toctree::
   :maxdepth: 1
   :caption: Spherical painting & lensing

   notebooks/07-Spherical_Painting_Methods
   notebooks/08-convergence-vs-glass

Resources
---------

* `Source on GitHub <https://github.com/DifferentiableUniverseInitiative/JaxPM>`_
* `jaxDecomp — distributed FFT & domain decomposition <https://github.com/DifferentiableUniverseInitiative/jaxDecomp>`_
