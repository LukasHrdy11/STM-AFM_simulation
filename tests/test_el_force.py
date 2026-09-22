"""Korektnostní test elektrostatické síly (`afm_sim/tip_force_el.py`, rov. 11.8).

Není to pytest test: skript vypíše [OK]/[SELHALO] pro každou kontrolu
a na konci souhrn (stejně jako test_vdw_sphere.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from afm_sim.tip_force import lj_force
from afm_sim.tip_force_el import EPS0, electrostatic_force, lj_plus_vdw_plus_el_force
from afm_sim.tip_force_vdw import lj_plus_vdw_force, vdw_sphere_force

U0 = 0.3 * 1.602e-19
Ra = 0.3e-9


def main():
    checks = []

    def check(description, ok, detail):
        checks.append(ok)
        print(f"[{'OK' if ok else 'SELHALO'}] {description}   ({detail})")

    # --- 1. příklad z knihy (kap. 11.1): R = 50 nm, z = 1 nm, V = 1 V -> ~1 nN
    f = electrostatic_force(1e-9, 1.0, 50e-9)
    check("kniha: R = 50 nm, z = 1 nm, V = 1 V dává ~1 nN (0,5-2 nN)",
          0.5e-9 < abs(f) < 2e-9, f"F = {f * 1e9:.3f} nN")

    # --- 2. vzorec -pi*eps0*V^2*R/D ------------------------------------------
    D, V, R = 3e-9, 0.7, 20e-9
    check("F_el = -pi*eps0*V^2*R/D",
          abs(electrostatic_force(D, V, R) / (-np.pi * EPS0 * V ** 2 * R / D) - 1.0) < 1e-12,
          "přesná shoda")

    # --- 3. přitažlivá, závisí na V^2 (znaménko V nehraje roli), klesá jako 1/D
    Ds = np.geomspace(0.3e-9, 100e-9, 40)
    check("F_el < 0 (přitažlivá) při libovolném V",
          bool(np.all(electrostatic_force(Ds, 1.3, 30e-9) < 0)
               and np.all(electrostatic_force(Ds, -1.3, 30e-9) < 0)), "V = ±1,3 V")
    check("F_el(V) = F_el(-V) a F_el(2V) = 4*F_el(V)",
          abs(electrostatic_force(D, -V, R) / electrostatic_force(D, V, R) - 1.0) < 1e-12
          and abs(electrostatic_force(D, 2 * V, R) / electrostatic_force(D, V, R) - 4.0) < 1e-12,
          "kvadratická závislost")
    check("F_el klesá s D jako 1/D (2 nm -> 4 nm = poloviční)",
          abs(electrostatic_force(4e-9, V, R) / electrostatic_force(2e-9, V, R) - 0.5) < 1e-12, "poměr 0,5")

    # --- 4. redukce na předchozí modely ---------------------------------------
    r = np.linspace(0.4e-9, 30e-9, 25)
    check("V = 0 -> shodné s lj_plus_vdw_force",
          bool(np.allclose(lj_plus_vdw_plus_el_force(r, U0, Ra, 1e-19, 50e-9, 0.0),
                           lj_plus_vdw_force(r, U0, Ra, 1e-19, 50e-9))), "25 bodů")
    check("V = 0 a H = 0 -> shodné s lj_force",
          bool(np.allclose(lj_plus_vdw_plus_el_force(r, U0, Ra, 0.0, 50e-9, 0.0),
                           lj_force(r, U0, Ra))), "25 bodů")
    check("součet: LJ + koule + elektrostatika",
          bool(np.allclose(lj_plus_vdw_plus_el_force(r, U0, Ra, 1e-19, 50e-9, 1.0),
                           lj_force(r, U0, Ra) + vdw_sphere_force(r, 1e-19, 50e-9)
                           + electrostatic_force(r, 1.0, 50e-9))), "25 bodů")

    print()
    if all(checks):
        print(f"Souhrn: {len(checks)}/{len(checks)} [OK]")
    else:
        print(f"Souhrn: {sum(checks)}/{len(checks)} [OK], {len(checks) - sum(checks)} [SELHALO]")


if __name__ == "__main__":
    main()
