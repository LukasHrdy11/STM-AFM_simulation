"""Korektnostní test Lennard-Jonesovy síly a posunu frekvence Δf(d).

Není to pytest test: skript vypíše [OK]/[SELHALO] pro každou kontrolu
a na konci souhrn (stejně jako test_kalibrace_kappa.py/test_sum.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from afm_sim.frequency_shift import frequency_shift
from afm_sim.tip_force import lj_force, lj_potential, r_min

U0 = 0.3 * 1.602e-19  # J
Ra = 0.3e-9           # m
K_CANT = 1800.0       # N/m
F0 = 30e3             # Hz


def main():
    checks = []

    def check(description, ok, detail):
        checks.append(ok)
        print(f"[{'OK' if ok else 'SELHALO'}] {description}   ({detail})")

    # --- 1. U(Ra) == 0 -----------------------------------------------------
    u_ra = lj_potential(Ra, U0, Ra)
    check("U(Ra) == 0", abs(u_ra) < 1e-30 * abs(U0) + 1e-40, f"U(Ra) = {u_ra:.3e} J")

    # --- 2. F(r_min) ~ 0 a správné znaménko po obou stranách ---------------
    rm = r_min(Ra)
    f_rm = lj_force(rm, U0, Ra)
    check("F(r_min) ≈ 0", abs(f_rm) < 1e-6 * U0 / Ra, f"F(r_min) = {f_rm:.3e} N")

    f_odpudive = lj_force(0.9 * rm, U0, Ra)
    f_privazlive = lj_force(1.5 * rm, U0, Ra)
    check("F(r < r_min) > 0 (odpudivá)", f_odpudive > 0, f"F = {f_odpudive:.3e} N")
    check("F(r > r_min) < 0 (přitažlivá)", f_privazlive < 0, f"F = {f_privazlive:.3e} N")

    # --- 3. malá amplituda: shoda s analytickou limitou -------------------
    force_fn = lambda r: lj_force(r, U0, Ra)
    d = 1.0e-9
    A_mala = 1.0e-14  # m, mnohonásobně menší než charakteristické délky
    h = 1e-15
    dF_dd = (force_fn(d + h) - force_fn(d - h)) / (2.0 * h)
    df_analyticky = F0 / (2.0 * K_CANT) * (-dF_dd)
    df_numericky = frequency_shift(d, A_mala, K_CANT, F0, force_fn)
    rel_chyba = abs(df_numericky / df_analyticky - 1.0)
    check("Δf(A -> 0) odpovídá analytické malé-amplitudové limitě (< 1 %)",
          rel_chyba < 0.01, f"{100 * rel_chyba:.4f} % rozdíl")

    # --- 4. nemonotónnost Δf(d): klesá, má minimum, pak roste -------------
    A = 0.3e-9
    grid = np.linspace(rm + A * 1.01, 5.0 * Ra, 200)
    vals = np.array([frequency_shift(d_, A, K_CANT, F0, force_fn) for d_ in grid])
    i_min = int(np.argmin(vals))
    ma_minimum_uvnitr = 0 < i_min < len(grid) - 1
    klesa_pred_minimem = vals[0] > vals[i_min]
    roste_za_minimem = vals[-1] > vals[i_min]
    check("Δf(d) má vnitřní minimum (nemonotónní, obr. 17.3)",
          ma_minimum_uvnitr, f"argmin na indexu {i_min}/{len(grid)}")
    check("Δf(d) klesá před minimem", klesa_pred_minimem,
          f"Δf(d_min_range) = {vals[0]:.3e} Hz > Δf(argmin) = {vals[i_min]:.3e} Hz")
    check("Δf(d) roste za minimem", roste_za_minimem,
          f"Δf(d_max_range) = {vals[-1]:.3e} Hz > Δf(argmin) = {vals[i_min]:.3e} Hz")

    print()
    if all(checks):
        print(f"Souhrn: {len(checks)}/{len(checks)} [OK]")
    else:
        print(f"Souhrn: {sum(checks)}/{len(checks)} [OK], "
              f"{len(checks) - sum(checks)} [SELHALO]")


if __name__ == "__main__":
    main()
