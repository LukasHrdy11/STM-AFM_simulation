"""Parametrický sken H / v / tau: minimum mezery na hraně a hranice nárazu.

Vždy s čistě integračním (I) regulátorem a okamžitým akčním členem
(T_sys = 0) - stejné zjednodušení jako v1.
"""

import numpy as np

from .controller import IntegralController, gain_from_tau
from .plant import FirstOrderPlant
from .sim import run_loop
from .surface import step_surface


def sweep_H_v(H_values, v_values, tau, x_edge, kappa, V, I_set, g_set,
              g_contact, dt_factor=100, settle_taus=20):
    """Pro pevné tau spočítá minimum mezery g_min a příznak nárazu na mřížce (H, v).

    Args:
        H_values: pole hodnot výšky schodu H [m].
        v_values: pole hodnot rychlosti pojezdu v [m/s].
        tau: časová konstanta I regulátoru [s].
        x_edge: poloha hrany schodu [m].
        kappa, V, I_set, g_set, g_contact: viz sim.run_loop.
        dt_factor: dt = tau / dt_factor.
        settle_taus: kolik násobků tau navíc simulovat po hraně (doba na doběhnutí).

    Returns:
        (g_min, crashed): 2D pole tvaru (len(H_values), len(v_values)).
        g_min[i, j] je minimální mezera během běhu pro H_values[i], v_values[j]
        (v metrech), crashed[i, j] je bool příznak nárazu.
    """
    dt = tau / dt_factor
    K_I = gain_from_tau(kappa, tau)
    g_min = np.empty((len(H_values), len(v_values)))
    crashed = np.empty((len(H_values), len(v_values)), dtype=bool)

    for i, H in enumerate(H_values):
        for j, v in enumerate(v_values):
            t_end = x_edge / v + settle_taus * tau
            surface_fn = lambda x, H=H: step_surface(x, x_edge, H)  # noqa: E731
            result = run_loop(
                controller=IntegralController(y0=g_set, K_I=K_I),
                plant=FirstOrderPlant(z0=g_set, T_sys=0.0),
                surface_fn=surface_fn,
                v=v, kappa=kappa, V=V, I_set=I_set, g_set=g_set,
                g_contact=g_contact, dt=dt, t_end=t_end,
            )
            g_min[i, j] = min(result.g)
            crashed[i, j] = result.crashed

    return g_min, crashed
