"""Lennard-Jonesova síla hrot-vzorek (kap. 11, Voigtländer, rovnice 11.5).

U(r) = 4*U0*[(Ra/r)^12 - (Ra/r)^6], F(r) = -dU/dr. Konvence: kladné F =
odpudivé (malé r), záporné F = přitažlivé (velké r) - viz obr. 11.1.
"""


def lj_potential(r, U0, Ra):
    """Lennard-Jonesův potenciál U(r) [J].

    Args:
        r: vzdálenost hrot-vzorek [m].
        U0: hloubka potenciálové jámy [J].
        Ra: vzdálenost, kde U(r) = 0 [m].
    """
    return 4.0 * U0 * ((Ra / r) ** 12 - (Ra / r) ** 6)


def lj_force(r, U0, Ra):
    """Lennard-Jonesova síla F(r) = -dU/dr [N] (kladné = odpudivé).

    Args:
        r: vzdálenost hrot-vzorek [m].
        U0: hloubka potenciálové jámy [J].
        Ra: vzdálenost, kde U(r) = 0 [m].
    """
    return 24.0 * U0 / r * (2.0 * (Ra / r) ** 12 - (Ra / r) ** 6)


def r_min(Ra):
    """Vzdálenost minima síly (hranice přitažlivá/odpudivá větev) [m].

    Odvození: dU/dr = 0 (mimo r = 0) dává (Ra/r)^6 = 1/2.
    """
    return 2.0 ** (1.0 / 6.0) * Ra
