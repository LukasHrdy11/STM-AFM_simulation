"""Tunelový proud bodového hrotu a odvozené veličiny setpointu."""

import math


def current(V, c, kappa, g):
    """Tunelový proud bodového hrotu.
    Args:
        V: předpětí [V].
        c: konstanta úměrnosti (viz c_from_setpoint) [A/V].
        kappa: rozpadová konstanta vlnové funkce [1/m].
        g: mezera hrot-vzorek [m].

    Returns:
        Tunelový proud I [A].
    """
    return V * c * math.exp(-2.0 * kappa * g)


def c_from_setpoint(V, kappa, g_set, I_set):
    """Dopočte konstantu c tak, aby při g = g_set vyšlo I = I_set.

    Inverzní funkce ke gap_setpoint.

    Args:
        V: předpětí [V].
        kappa: rozpadová konstanta vlnové funkce [1/m].
        g_set: zvolená nominální mezera hrot-vzorek [m].
        I_set: požadovaný proud (setpoint) [A].

    Returns:
        Konstanta c [A/V].
    """
    return I_set / (V * math.exp(-2.0 * kappa * g_set))


def gap_setpoint(V, c, kappa, I_set):
    """Nominální mezera g_set, při které (pro dané V, c, kappa) vyjde I = I_set.

    Inverzní funkce k c_from_setpoint.

    Args:
        V: předpětí [V].
        c: konstanta úměrnosti (viz c_from_setpoint) [A/V].
        kappa: rozpadová konstanta vlnové funkce [1/m].
        I_set: požadovaný proud (setpoint) [A].

    Returns:
        Nominální mezera g_set [m].
    """
    return math.log(V * c / I_set) / (2.0 * kappa)
