"""Rozšíření síly hrot-vzorek o van der Waalsův příspěvek celého hrotu.

Doplněk k `afm_sim/tip_force.py` - ten se NEMĚNÍ a zůstává výchozí. Lennard-
Jones tam popisuje jediný pár atomů (apex hrotu - nejbližší atom vzorku),
takže jeho přitažlivá větev klesá jako 1/r^7. Voigtländer (kap. 11.1, rov.
11.2-11.4) ale upozorňuje, že kvůli dlouhému dosahu -C/r^6 se musí sečíst
příspěvky atomů z celého objemu hrotu i vzorku. Sečtením (integrací přes
objemy) dostaneme mnohem pomaleji klesající sílu.

Tady je to řešeno geometrií koule (hrot) nad půlprostorem (vzorek):

    U_vdW(D) = -(H/6) * [R/D + R/(D + 2R) + ln(D/(D + 2R))]
    F_vdW(D) = -dU/dD = -2*H*R^3 / (3*D^2*(D + 2R)^2)

Pro D << R přechází na Voigtländrovu rov. 11.3/11.4, U = -H*R/(6*D),
F = -H*R/(6*D^2); pro D >> R klesá síla jen jako 1/D^4 (koule se chová
jako jeden bod). Konvence znaménka stejná jako `lj_force`: kladné F =
odpudivé, záporné = přitažlivé.

`D` je vzdálenost apexu hrotu od povrchu vzorku (stejná proměnná `r`, kterou
dostává `lj_force`), `H` je Hamakerova konstanta [J], `R` poloměr hrotu [m].

Pozor na dvojí započtení: `lj_plus_vdw_force` přičítá kulovou sílu k LJ, jehož
vlastní -1/r^6 člen už interakci apexu zahrnuje. Jde o hrubou superpozici
(LJ = krátkodosahová část z nejbližších atomů, koule = dlouhodosahové pozadí),
ne o dvě nezávislé síly. U vzdáleností několika nm je LJ chvost zanedbatelný
proti kulovému členu, takže tam se chyba neprojeví; blízko r_min záleží na H, R.
"""

import numpy as np

from afm_sim.tip_force import lj_force


def vdw_sphere_potential(D, H, R):
    """Van der Waalsova energie koule (poloměr R) nad půlprostorem [J].

    Args:
        D: vzdálenost apexu od povrchu [m].
        H: Hamakerova konstanta [J].
        R: poloměr hrotu [m].
    """
    return -(H / 6.0) * (R / D + R / (D + 2.0 * R) + np.log(D / (D + 2.0 * R)))


def vdw_sphere_force(D, H, R):
    """Síla koule nad půlprostorem F = -dU/dD [N] (záporná = přitažlivá)."""
    return -2.0 * H * R ** 3 / (3.0 * D ** 2 * (D + 2.0 * R) ** 2)


def vdw_sphere_force_limit(D, H, R):
    """Limita D << R: F = -H*R/(6*D^2) (Voigtländer rov. 11.4) [N]."""
    return -H * R / (6.0 * D ** 2)


def lj_plus_vdw_force(r, U0, Ra, H, R):
    """Lennard-Jonesova síla (jeden pár atomů) + van der Waalsova síla koule [N].

    Rozšíření modelu - s H = 0 (nebo R = 0) je shodná s `lj_force`.
    """
    return lj_force(r, U0, Ra) + vdw_sphere_force(r, H, R)
