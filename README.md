# JaxPM
[![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DifferentiableUniverseInitiative/JaxPM/blob/main/notebooks/01-Introduction.ipynb)
[![PyPI version](https://img.shields.io/pypi/v/jaxpm)](https://pypi.org/project/jaxpm/) [![Tests](https://github.com/DifferentiableUniverseInitiative/JaxPM/actions/workflows/tests.yml/badge.svg)](https://github.com/DifferentiableUniverseInitiative/JaxPM/actions/workflows/tests.yml) <!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
JAX-powered Cosmological Particle-Mesh N-body Solver

> ### Note
> **The new JaxPM v0.1.xx** supports multi-GPU model distribution while remaining compatible with previous releases. These significant changes are still under development and testing, so please report any issues you encounter.
> For the older but more stable version, install:
> ```bash
> pip install jaxpm==0.0.2
> ```

## Install

Basic installation can be done using pip:
```bash
pip install jaxpm
```
For more advanced installation for optimized distribution on gpu clusters, please install jaxDecomp first. See instructions [here](https://github.com/DifferentiableUniverseInitiative/jaxDecomp).


## Goals

Provide a modern infrastructure to support differentiable PM N-body simulations using JAX:
- Keep implementation simple and readable, in pure NumPy API
- Any order forward and backward automatic differentiation
- Support automated batching using `vmap`
- Compatibility with external optimizer libraries like `optax`
- Now fully distributable on **multi-GPU and multi-node** systems using [jaxDecomp](https://github.com/DifferentiableUniverseInitiative/jaxDecomp) working with`JAX v0.4.35`


## Open development and use

Current expectations are:
- This project is and will remain open source, and usable without any restrictions for any purposes
- Will be a simple publication on [The Journal of Open Source Software](https://joss.theoj.org/)
- Everyone is welcome to contribute, and can join the JOSS publication (until it is submitted to the journal).
- Anyone (including main contributors) can use this code as a framework to build and publish their own applications, with no expectation that they *need* to extend authorship to all jaxpm developers.

## Getting Started

To dive into JaxPM’s capabilities, please explore the **notebook section** for detailed tutorials and examples on various setups, from single-device simulations to multi-host configurations. You can find the notebooks' [README here](notebooks/README.md) for a structured guide through each tutorial.


## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://flanusse.net"><img src="https://avatars.githubusercontent.com/u/861591?v=4?s=100" width="100px;" alt="Francois Lanusse"/><br /><sub><b>Francois Lanusse</b></sub></a><br /><a href="#ideas-EiffL" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dlanzieri"><img src="https://avatars.githubusercontent.com/u/72620117?v=4?s=100" width="100px;" alt="Denise Lanzieri"/><br /><sub><b>Denise Lanzieri</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/JaxPM/commits?author=dlanzieri" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ASKabalan"><img src="https://avatars.githubusercontent.com/u/83787080?v=4?s=100" width="100px;" alt="Wassim KABALAN"/><br /><sub><b>Wassim KABALAN</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/JaxPM/commits?author=ASKabalan" title="Code">💻</a> <a href="#infra-ASKabalan" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/DifferentiableUniverseInitiative/JaxPM/pulls?q=is%3Apr+reviewed-by%3AASKabalan" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/hsimonfroy"><img src="https://avatars.githubusercontent.com/u/85559558?v=4?s=100" width="100px;" alt="Hugo Simon-Onfroy"/><br /><sub><b>Hugo Simon-Onfroy</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/JaxPM/commits?author=hsimonfroy" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://aboucaud.github.io"><img src="https://avatars.githubusercontent.com/u/3065310?v=4?s=100" width="100px;" alt="Alexandre Boucaud"/><br /><sub><b>Alexandre Boucaud</b></sub></a><br /><a href="https://github.com/DifferentiableUniverseInitiative/JaxPM/pulls?q=is%3Apr+reviewed-by%3Aaboucaud" title="Reviewed Pull Requests">👀</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
