import jax.numpy as np
from jax.numpy import interp
# Import all background cosmology functions from jax_cosmo
# This includes: growth_factor, growth_rate, growth_factor_second, growth_rate_second,
# Esqr, Omega_m_a, Omega_de_a, w, f_de, H, and distance functions
from jax_cosmo.background import (Esqr, Omega_de_a, Omega_m_a,
                                  _compute_growth_tables, f_de, growth_factor,
                                  growth_factor_second, growth_rate,
                                  growth_rate_second, w)

__all__ = [
    # Re-exported from jax_cosmo
    "growth_factor",
    "growth_rate",
    "growth_factor_second",
    "growth_rate_second",
    # JaxPM-specific functions
    "E",
    "df_de",
    "dEa",
    "Gf",
    "Gf2",
    "dGfa",
    "dGf2a",
    "gp",
    "Dplusdada",
    "Dplus_to_a",
]


def E(cosmo, a):
    r"""Scale factor dependent factor E(a) in the Hubble parameter.

    Parameters
    ----------
    cosmo: Cosmology
        Cosmological parameters structure

    a : array_like
        Scale factor

    Returns
    -------
    E : ndarray, or float if input scalar
        The scaling of the Hubble constant as a function of scale factor

    Notes
    -----
    The Hubble parameter at scale factor `a` is given by
    :math:`H(a) = E(a) H_0` where :math:`E(a) = \sqrt{E^2(a)}`.
    """
    return np.sqrt(Esqr(cosmo, a))


def df_de(cosmo, a, epsilon=1e-5):
    r"""Derivative of the Dark Energy evolution parameter f(a) with respect
    to the scale factor.

    Parameters
    ----------
    cosmo: Cosmology
        Cosmological parameters structure

    a : array_like
        Scale factor

    epsilon: float
        Small number to avoid singularity at a=1, default 1e-5

    Returns
    -------
    df/da : ndarray, or float if input scalar
        Derivative of the Dark Energy evolution parameter

    Notes
    -----
    For the Linder parametrization, the derivative is:

    .. math::

        \frac{df}{da}(a) = \frac{3 w_a \left( \ln(a-\epsilon) -
        \frac{a-1}{a-\epsilon} \right)}{\ln^2(a-\epsilon)}
    """
    return (3 * cosmo.wa * (np.log(a - epsilon) - (a - 1) / (a - epsilon)) /
            np.power(np.log(a - epsilon), 2))


def dEa(cosmo, a):
    r"""Derivative of E(a) with respect to the scale factor.

    Parameters
    ----------
    cosmo: Cosmology
        Cosmological parameters structure

    a : array_like
        Scale factor

    Returns
    -------
    dE/da : ndarray, or float if input scalar
        Derivative of E(a) with respect to scale factor

    Notes
    -----
    The expression for :math:`\frac{dE}{da}` is:

    .. math::

        \frac{dE}{da} = \frac{-3 \Omega_m a^{-4} - 2 \Omega_k a^{-3}
        + f'_{de} \Omega_{de} e^{f_{de}(a)}}{2 E(a)}
    """
    return (0.5 * (-3 * cosmo.Omega_m * np.power(a, -4) -
                   2 * cosmo.Omega_k * np.power(a, -3) +
                   df_de(cosmo, a) * cosmo.Omega_de * np.exp(f_de(cosmo, a))) /
            E(cosmo, a))


# =============================================================================
# FastPM Growth Functions
# =============================================================================


def Gf(cosmo, a):
    r"""FastPM first-order growth factor function.

    Parameters
    ----------
    cosmo: Cosmology
        Cosmology object

    a : array_like
        Scale factor

    Returns
    -------
    Gf : ndarray, or float if input scalar
        FastPM growth factor function

    Notes
    -----
    The expression for :math:`G_f(a)` is:

    .. math::

        G_f(a) = D'_1 \cdot a^3 \cdot E(a)

    where :math:`D'_1 = dD_1/da` is the derivative of the first-order
    growth factor.
    """
    f1 = growth_rate(cosmo, a)
    g1 = growth_factor(cosmo, a)
    D1f = f1 * g1 / a  # dD1/da = f1 * D1 / a
    return D1f * np.power(a, 3) * E(cosmo, a)


def Gf2(cosmo, a):
    r"""FastPM second-order growth factor function.

    Parameters
    ----------
    cosmo: Cosmology
        Cosmology object

    a : array_like
        Scale factor

    Returns
    -------
    Gf2 : ndarray, or float if input scalar
        FastPM second-order growth factor function

    Notes
    -----
    The expression for :math:`G_{f2}(a)` is:

    .. math::

        G_{f2}(a) = D'_2 \cdot a^3 \cdot E(a)

    where :math:`D'_2 = dD_2/da` is the derivative of the second-order
    growth factor.
    """
    f2 = growth_rate_second(cosmo, a)
    g2 = growth_factor_second(cosmo, a)
    D2f = f2 * g2 / a  # dD2/da = f2 * D2 / a
    return D2f * np.power(a, 3) * E(cosmo, a)


def dGfa(cosmo, a):
    r"""Derivative of Gf with respect to scale factor.

    Parameters
    ----------
    cosmo: Cosmology
        Cosmology object

    a : array_like
        Scale factor

    Returns
    -------
    dGf/da : ndarray, or float if input scalar
        Derivative of Gf with respect to scale factor

    Notes
    -----
    The expression for :math:`\frac{dG_f}{da}` is:

    .. math::

        \frac{dG_f}{da} = D''_1 a^3 E(a) + D'_1 a^3 E'(a) + 3 a^2 E(a) D'_1
    """
    f1 = growth_rate(cosmo, a)
    g1 = growth_factor(cosmo, a)
    D1f = f1 * g1 / a

    cache = _compute_growth_tables(cosmo)
    # cache tuple: (atab, gtab, ftab, htab, g2tab, f2tab, h2tab)
    f1p = cache[3] / cache[0] * cache[1]
    f1p = interp(np.log(a), np.log(cache[0]), f1p)

    Ea = E(cosmo, a)
    return f1p * a**3 * Ea + D1f * a**3 * dEa(cosmo, a) + 3 * a**2 * Ea * D1f


def dGf2a(cosmo, a):
    r"""Derivative of Gf2 with respect to scale factor.

    Parameters
    ----------
    cosmo: Cosmology
        Cosmology object

    a : array_like
        Scale factor

    Returns
    -------
    dGf2/da : ndarray, or float if input scalar
        Derivative of Gf2 with respect to scale factor

    Notes
    -----
    The expression for :math:`\frac{dG_{f2}}{da}` is:

    .. math::

        \frac{dG_{f2}}{da} = D''_2 a^3 E(a) + D'_2 a^3 E'(a) + 3 a^2 E(a) D'_2
    """
    f2 = growth_rate_second(cosmo, a)
    g2 = growth_factor_second(cosmo, a)
    D2f = f2 * g2 / a

    cache = _compute_growth_tables(cosmo)
    # cache tuple: (atab, gtab, ftab, htab, g2tab, f2tab, h2tab)
    f2p = cache[6] / cache[0] * cache[4]
    f2p = interp(np.log(a), np.log(cache[0]), f2p)

    Ea = E(cosmo, a)
    return f2p * a**3 * Ea + D2f * a**3 * dEa(cosmo, a) + 3 * a**2 * Ea * D2f


def gp(cosmo, a):
    r"""Derivative of the first-order growth factor D1 with respect to
    scale factor.

    Parameters
    ----------
    cosmo: Cosmology
        Cosmology object

    a : array_like
        Scale factor

    Returns
    -------
    dD1/da : ndarray, or float if input scalar
        Derivative of D1 with respect to scale factor

    Notes
    -----
    The expression for :math:`g_p(a)` is:

    .. math::

        g_p(a) = \frac{dD_1}{da} = \frac{f_1 \cdot D_1}{a}

    where :math:`f_1 = d\ln D_1 / d\ln a` is the growth rate.
    """
    f1 = growth_rate(cosmo, a)
    g1 = growth_factor(cosmo, a)
    return f1 * g1 / a


def Dplusdada(cosmo, a):
    r"""Second derivative of the first-order growth factor D1 with respect to scale factor.

    Parameters
    ----------
    cosmo: Cosmology
        Cosmology object

    a : array_like
        Scale factor

    Returns
    -------
    d²D1/da² : ndarray, or float if input scalar
    """
    cache = _compute_growth_tables(cosmo)
    # cache[3] = htab = a·D''/D, so D'' = htab·gtab/atab
    dplusdada_tab = cache[3] * cache[1] / cache[0]
    return interp(np.log(np.atleast_1d(a)), np.log(cache[0]), dplusdada_tab)


def Dplus_to_a(cosmo, D):
    r"""Inverse growth factor: given D+(a), return a.

    Parameters
    ----------
    cosmo: Cosmology
        Cosmology object

    D : array_like
        Growth factor value(s)

    Returns
    -------
    a : ndarray, or float if input scalar
        Scale factor corresponding to the given growth factor
    """
    cache = _compute_growth_tables(cosmo)
    return interp(np.atleast_1d(D), cache[1], cache[0])
