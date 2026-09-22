"""Posun frekvence Δf(d) pro FM-AFM (kap. 17, Voigtländer, rovnice 17.15) a
z něj odvozená efektivní citlivost smyčky.

Δf(d) je na rozdíl od STM I(g) NEMONOTÓNNÍ funkce vzdálenosti d - klesá
v přitažlivé oblasti, má lokální minimum na hranici přitažlivá/odpudivá
(viz afm_sim.tip_force.r_min), pak roste. Pracovní bod d_set musí ležet
na monotónní (přitažlivé) větvi PŘED tímto minimem - jinak by regulátor
viděl obrácené znaménko citlivosti (viz sanity check v run_simulation_afm.py).
"""

import numpy as np


def frequency_shift(d, A, k_cant, f0, force_fn, n_theta=400):
    """Posun frekvence Δf(d) váženým průměrem síly přes oscilační cyklus.

    Přesná rovnice 17.15 (platí pro libovolnou amplitudu A, ne jen limitu
    velké amplitudy):

        Δf(d) = -(f0 / (π*k*A²)) * ∫_{-A}^{A} F(d+z) * z/√(A²-z²) dz

    Substitucí z = A*sinθ singularita na krajích integrálu zmizí:

        Δf(d) = -(f0 / (π*k*A)) * ∫_{-π/2}^{π/2} F(d + A*sinθ) * sinθ dθ

    Integruje se numericky (numpy.trapz) na jemné mřížce θ.

    Args:
        d: střední (rovnovážná) poloha oscilace, ne okamžitá výchylka -
            ta je uvnitř integrálu d + A*sinθ. Dolní úvrať cyklu je
            d - A, což je vzdálenost, která musí zůstat nad hranicí
            náraz/kontakt (viz d_contact v run_simulation_afm.py) [m].
        A: amplituda oscilace cantileveru [m].
        k_cant: tuhost cantileveru [N/m].
        f0: rezonanční frekvence cantileveru [Hz].
        force_fn: funkce F(r) -> síla hrot-vzorek [N] (viz tip_force.lj_force).
        n_theta: počet bodů mřížky θ pro numerickou integraci.

    Returns:
        Posun frekvence Δf [Hz].
    """
    theta = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_theta)
    r = d + A * np.sin(theta)
    integrand = force_fn(r) * np.sin(theta)
    integral = np.trapz(integrand, theta)
    return -(f0 / (np.pi * k_cant * A)) * integral


def kappa_eff(d, A, k_cant, f0, force_fn, h=1e-13, n_theta=400):
    """Efektivní citlivost kappa_eff(d) = -de/dd = dΔf/dd (centrální diference).

    Chyba e = df_set - Δf(d) už nese znaménko obrácené vůči Δf, takže
    -de/dd = +dΔf/dd (bez vlastního minus, na rozdíl od prvního dojmu
    z analogie se STM, kde e = ln(I/I_set) samo klesá s g). Na
    použitelné (přitažlivé, monotónní) větvi je Δf rostoucí funkcí d,
    takže tam kappa_eff > 0 - stejně jako STM kappa. Bez faktoru 2, ten
    u STM pramenil konkrétně z exp(-2*kappa*g), tady žádný takový
    faktor není.

    Args:
        d, A, k_cant, f0, force_fn, n_theta: viz frequency_shift.
        h: krok centrální diference [m].

    Returns:
        kappa_eff [Hz/m].
    """
    df_plus = frequency_shift(d + h, A, k_cant, f0, force_fn, n_theta)
    df_minus = frequency_shift(d - h, A, k_cant, f0, force_fn, n_theta)
    return (df_plus - df_minus) / (2.0 * h)


def gain_from_tau(kappa_eff_value, tau):
    """Zesílení K_I odpovídající zvolené časové konstantě I-smyčky AFM.

    K_I = 1 / (kappa_eff * tau)

    Na rozdíl od stm_sim.controller.gain_from_tau tu chybí faktor 2 -
    stejný důvod jako u kappa_eff (viz výše).

    Args:
        kappa_eff_value: efektivní citlivost v pracovním bodě [Hz/m].
        tau: požadovaná časová konstanta smyčky [s].

    Returns:
        Zesílení K_I [m/s na jednotku chyby e v Hz].
    """
    return 1.0 / (kappa_eff_value * tau)
