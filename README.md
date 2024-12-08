# JaxPM
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
JAX-powered Cosmological Particle-Mesh N-body Solver

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
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
