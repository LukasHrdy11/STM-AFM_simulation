"""Korektnostní test fázového posunu φ(d) - Cesta A (resonance_phase), Cesta B
(phase_shift_numeric) a Cesta C (pll_phase_shift), viz afm_sim/phase_shift.py.

Není to pytest test: skript vypíše [OK]/[SELHALO] pro každou kontrolu
a na konci souhrn (stejně jako test_afm_frequency_shift.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from afm_sim.frequency_shift import frequency_shift
from afm_sim.phase_shift import resonance_phase, phase_shift_numeric, pll_phase_shift
from afm_sim.tip_force import lj_force, r_min

U0 = 0.3 * 1.602e-19  # J
Ra = 0.3e-9           # m
K_CANT = 1800.0       # N/m
F0 = 30e3             # Hz
Q = 300.0             # vzduchové Q, typické pro AM mód (kap. 14.5)


def main():
    checks = []

    def check(description, ok, detail):
        checks.append(ok)
        print(f"[{'OK' if ok else 'SELHALO'}] {description}   ({detail})")

    force_fn = lambda r: lj_force(r, U0, Ra)
    rm = r_min(Ra)
    d = 1.0e-9

    # --- 1. resonance_phase (Cesta A) v limitě malé amplitudy odpovídá (14.16) ---
    A_mala = 1.0e-14  # m
    h = 1e-15
    dF_dd = (force_fn(d + h) - force_fn(d - h)) / (2.0 * h)
    df_analyticky = F0 / (2.0 * K_CANT) * (-dF_dd)
    # Δφ = -(Q/k)*F' (14.16) a Δf = -(f0/2k)*F' (17.15 v malé amplitudě) =>
    # F' = -2k*Δf/f0 => Δφ = +(2Q/f0)*Δf (POZOR na znaménko oproti naivnímu
    # očekávání - ověřeno dosazením do (14.16)/(17.15), viz diskuze v kapitole).
    dphi_analyticky = (2.0 * Q / F0) * df_analyticky
    phi_analyticky = -np.pi / 2.0 + dphi_analyticky

    phi_a = resonance_phase(d, A_mala, K_CANT, F0, Q, force_fn)
    rel_chyba_a = abs(phi_a - phi_analyticky) / abs(dphi_analyticky)
    check("resonance_phase(A -> 0) odpovídá lineární limitě (14.16), < 1 %",
          rel_chyba_a < 0.01, f"φ_A={np.degrees(phi_a):.6f}°, "
          f"φ_analyt={np.degrees(phi_analyticky):.6f}°")

    # --- 2. phase_shift_numeric (Cesta B) v limitě malé amplitudy odpovídá (14.16) ---
    # Mírně větší amplituda než A_mala (ta je zvolená jen pro přesnou centrální
    # diferenci síly) - u numerické integrace by extrémně malá amplituda vedla
    # na fázový posun srovnatelný s numerickým šumem RK4/plovoucí čárky.
    A_num_mala = 3.0e-12  # m
    dF_dd_num = (force_fn(d + h) - force_fn(d - h)) / (2.0 * h)
    df_analyticky_num = F0 / (2.0 * K_CANT) * (-dF_dd_num)
    dphi_analyticky_num = (2.0 * Q / F0) * df_analyticky_num
    phi_analyticky_num = -np.pi / 2.0 + dphi_analyticky_num

    _, phi_b = phase_shift_numeric(d, A_num_mala, K_CANT, F0, Q, force_fn,
                                    n_periods_transient=int(3 * Q), n_periods_measure=20)
    rel_chyba_b = abs(phi_b - phi_analyticky_num) / abs(dphi_analyticky_num)
    check("phase_shift_numeric(A malé) odpovídá lineární limitě (14.16), < 10 %",
          rel_chyba_b < 0.10, f"φ_B={np.degrees(phi_b):.6f}°, "
          f"φ_analyt={np.degrees(phi_analyticky_num):.6f}°")

    A_meas_check, _ = phase_shift_numeric(d, A_num_mala, K_CANT, F0, Q, force_fn,
                                           n_periods_transient=int(3 * Q), n_periods_measure=20)
    check("phase_shift_numeric vrací A_meas blízké zadané cílové amplitudě",
          abs(A_meas_check / A_num_mala - 1.0) < 0.05,
          f"A_meas={A_meas_check:.3e} m, zadáno {A_num_mala:.3e} m")

    # --- 3. Cesta A vs Cesta B si navzájem odpovídají pro reprezentativní A ---
    A_repr = 2.0e-11  # m, dost malá na malo-amplitudové rozhraní, ale ne nekonečně
    d_repr = 1.2e-9
    phi_a_repr = resonance_phase(d_repr, A_repr, K_CANT, F0, Q, force_fn)
    _, phi_b_repr = phase_shift_numeric(d_repr, A_repr, K_CANT, F0, Q, force_fn,
                                         n_periods_transient=int(3 * Q),
                                         n_periods_measure=20)
    rozdil_stupne = abs(np.degrees(phi_a_repr) - np.degrees(phi_b_repr))
    check("Cesta A a Cesta B si odpovídají pro reprezentativní amplitudu (< 2°)",
          rozdil_stupne < 2.0,
          f"φ_A={np.degrees(phi_a_repr):.3f}°, φ_B={np.degrees(phi_b_repr):.3f}°")

    # --- 4. φ zůstává v rozumném rozsahu blízko -90° na monotónní větvi ------
    d_mono = 1.5 * rm
    phi_mono = resonance_phase(d_mono, A_mala, K_CANT, F0, Q, force_fn)
    v_rozsahu = -np.pi < phi_mono < 0.0
    check("φ na monotónní větvi leží v (-180°, 0°)", v_rozsahu,
          f"φ = {np.degrees(phi_mono):.3f}°")

    # --- 5. Cesta C (PLL) zamkne na Δf odpovídající Cestě A ------------------
    # Blíž ke vzorku než d_repr - potřebujeme Δf řádu jednotek Hz (ne desetin),
    # aby relativní odchylka nebyla dominovaná numerickým šumem RK4 na téměř
    # nulové hodnotě.
    A_pll = 2.0e-11  # m
    d_pll = 0.6e-9   # m
    df_pll, phi_pll, locked = pll_phase_shift(d_pll, A_pll, K_CANT, F0, Q, force_fn)
    df_ocekavane = frequency_shift(d_pll, A_pll, K_CANT, F0, force_fn)
    check("pll_phase_shift zamkne (locked == True) na bezpečné větvi",
          locked, f"locked={locked}")
    rel_chyba_pll = abs(df_pll - df_ocekavane) / abs(df_ocekavane)
    check("pll_phase_shift: df_pll odpovídá frequency_shift (Cesta A), < 1 %",
          rel_chyba_pll < 0.01,
          f"df_pll={df_pll:.3f} Hz, frequency_shift={df_ocekavane:.3f} Hz")

    # --- 6. Cesta C: fáze u zamčeného PLL je blízko 0 (nenese info o d) -----
    check("pll_phase_shift: phi_pll je blízko 0 (fáze u zámku nenese info o d)",
          abs(phi_pll) < 1e-2,
          f"phi_pll={np.degrees(phi_pll):.4f}°")

    # --- 7. Cesta C: locked == False při nedostatečně dlouhé simulaci -------
    # Ověřuje, že "locked" skutečně něco rozlišuje (ne vždy True) - výchozí
    # smyčka se ustálí až za ~1,6 ms, 0,1 ms na to nestačí (e sotva odejde
    # od výchozích -90°).
    _, _, locked_kratce = pll_phase_shift(d_pll, A_pll, K_CANT, F0, Q, force_fn,
                                           t_end=1e-4)
    check("pll_phase_shift: locked == False při nedostatečné době ustálení",
          not locked_kratce, "t_end=0,1 ms << doba ustálení výchozí smyčky (~1,6 ms)")

    print()
    if all(checks):
        print(f"Souhrn: {len(checks)}/{len(checks)} [OK]")
    else:
        print(f"Souhrn: {sum(checks)}/{len(checks)} [OK], "
              f"{len(checks) - sum(checks)} [SELHALO]")


if __name__ == "__main__":
    main()
