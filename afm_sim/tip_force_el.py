"""Rozšíření síly hrot-vzorek o elektrostatický příspěvek (Voigtländer, rov. 11.8).

Doplněk k `afm_sim/tip_force.py` (LJ) a `afm_sim/tip_force_vdw.py` (vdW koule),
které se NEMĚNÍ. Hrot a vzorek tvoří kondenzátor s kapacitou C(z); síla je
F = -1/2 * dC/dz * V^2 (rov. 11.6). Pro hrot jako kouli na kuželu nad
půlprostorem kniha uvádí přibližně (rov. 11.8):

    F_el(D) = -pi * eps0 * V^2 * R / D

`D` je vzdálenost apexu od povrchu (stejná proměnná `r` jako u `lj_force`),
`R` poloměr apexu [m], `V` efektivní potenciálový rozdíl [V]. Znaménko jako
u ostatních sil: záporné = přitažlivé (F_el je přitažlivá při jakémkoli V).

`V = V_bias - Phi/e` (kap. 11.1): rozdíl výstupních prací hrot-vzorek (kontaktní
potenciál) posouvá minimum paraboly F_el(V_bias) od nuly, takže i při
V_bias = 0 je V obecně nenulové. Samotné V_bias tedy V neurčuje.

Vzorec je aproximace pro D << R; nepočítá s kuželem a raménkem, které
u velkých D přispívají také (kniha to zmiňuje výslovně).
"""

import numpy as np

from afm_sim.tip_force_vdw import lj_plus_vdw_force

EPS0 = 8.8541878128e-12  # F/m


def electrostatic_force(D, V, R):
    """Elektrostatická síla hrotu nad vodivým vzorkem [N] (záporná = přitažlivá)."""
    return -np.pi * EPS0 * V ** 2 * R / D


def lj_plus_vdw_plus_el_force(r, U0, Ra, H, R, V):
    """LJ + van der Waalsova koule + elektrostatika [N].

    Rozšíření modelu: s V = 0 je shodná s `lj_plus_vdw_force`, s H = 0 a V = 0
    s `lj_force`. `R` je společný poloměr apexu pro vdW i elektrostatický člen.
    """
    return lj_plus_vdw_force(r, U0, Ra, H, R) + electrostatic_force(r, V, R)
