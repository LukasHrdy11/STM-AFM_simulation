"""Korektnostní test rozšíření o van der Waalsovu sílu koule (`afm_sim/tip_force_vdw.py`).

Není to pytest test: skript vypíše [OK]/[SELHALO] pro každou kontrolu
a na konci souhrn (stejně jako test_afm_frequency_shift.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from afm_sim.frequency_shift import frequency_shift
from afm_sim.tip_force import lj_force
from afm_sim.tip_force_vdw import (lj_plus_vdw_force, vdw_sphere_force,
                                   vdw_sphere_force_limit, vdw_sphere_potential)

H = 1.0e-19     # J
R = 20e-9       # m
U0 = 0.3 * 1.602e-19
Ra = 0.3e-9
K_CANT, F0 = 1800.0, 30e3


def _energie_soucet_atomu(D, H, R, n=4000):
    """NEZÁVISLÝ výpočet energie: sečte -C*rho^2/r^6 přes atomy koule a
    půlprostoru přímou numerickou integrací (bez použití uzavřeného vzorce).

    Pro atom ve výšce z nad půlprostorem dá integrál přes půlprostor
    (válcové souřadnice) energii -C*rho_B*pi/(6 z^3). Koule se rozřeže na
    tenké disky kolmé na osu, každý disk (poloměr sqrt(R^2 - (z-D-R)^2)) je
    množina atomů ve stejné výšce. H = pi^2*C*rho_A*rho_B, takže vyjde
    U = -(H/(6*pi)) * integral(pi*(R^2 - (z-D-R)^2) / z^3 dz).
    """
    # Krok 1: ověř formuli půlprostoru numerickou 2D integrací pro jeden atom.
    z0 = 1.3e-9
    rho_max, zmax = 400e-9, 400e-9
    rr = np.linspace(0.0, rho_max, n)
    zz = np.linspace(z0, zmax, n)
    Rr, Zz = np.meshgrid(rr, zz)
    integrand = 2.0 * np.pi * Rr / (Rr ** 2 + Zz ** 2) ** 3   # 1/r^6, dV = 2*pi*rho drho dz
    pul = np.trapz(np.trapz(integrand, rr, axis=1), zz)
    pul_analyticky = np.pi / (6.0 * z0 ** 3)

    # Krok 2: sečti disky koule.
    z = np.linspace(D, D + 2.0 * R, 200001)
    plocha = np.pi * np.clip(R ** 2 - (z - D - R) ** 2, 0.0, None)
    integral = np.trapz(plocha / z ** 3, z)
    return -(H / (6.0 * np.pi)) * integral, pul / pul_analyticky


def main():
    checks = []

    def check(description, ok, detail):
        checks.append(ok)
        print(f"[{'OK' if ok else 'SELHALO'}] {description}   ({detail})")

    # --- 1. limita D << R = Voigtländer rov. 11.4 ---------------------------
    D = 0.2e-9
    f, f_lim = vdw_sphere_force(D, H, R), vdw_sphere_force_limit(D, H, R)
    check("F_sphere(D << R) = -H*R/(6*D^2) (rov. 11.4), do 1 %",
          abs(f / f_lim - 1.0) < 0.01, f"{100 * (f / f_lim - 1):+.3f} %")

    # --- 2. přitažlivá (záporná) ve všech vzdálenostech ----------------------
    Ds = np.geomspace(0.1e-9, 500e-9, 50)
    check("F_sphere < 0 (přitažlivá) všude", bool(np.all(vdw_sphere_force(Ds, H, R) < 0)),
          "50 vzdáleností 0,1-500 nm")

    # --- 3. F = -dU/dD --------------------------------------------------------
    D = 3e-9
    h = 1e-13
    f_num = -(vdw_sphere_potential(D + h, H, R) - vdw_sphere_potential(D - h, H, R)) / (2 * h)
    check("F_sphere = -dU/dD (numerická derivace potenciálu)",
          abs(f_num / vdw_sphere_force(D, H, R) - 1.0) < 1e-4,
          f"{100 * (f_num / vdw_sphere_force(D, H, R) - 1):+.5f} %")

    # --- 4. nezávislé sečtení atomů -------------------------------------------
    U_soucet, pomer_pul = _energie_soucet_atomu(3e-9, H, R)
    check("půlprostor: numerický integrál 1/r^6 = pi/(6 z^3), do 1 %",
          abs(pomer_pul - 1.0) < 0.01, f"poměr {pomer_pul:.5f}")
    U_vzorec = vdw_sphere_potential(3e-9, H, R)
    check("U koule: součet disků atomů = uzavřený vzorec, do 0,1 %",
          abs(U_soucet / U_vzorec - 1.0) < 1e-3, f"{100 * (U_soucet / U_vzorec - 1):+.4f} %")

    # --- 5. pomalejší pokles než LJ ------------------------------------------
    r1, r2 = 2e-9, 4e-9
    pomer_lj = lj_force(r1, U0, Ra) / lj_force(r2, U0, Ra)
    pomer_vdw = vdw_sphere_force(r1, H, R) / vdw_sphere_force(r2, H, R)
    check("koule klesá s D pomaleji než LJ (2 nm -> 4 nm)", pomer_vdw < pomer_lj / 10,
          f"pokles ×{pomer_vdw:.1f} vs. ×{pomer_lj:.0f} u LJ")

    # --- 6. H = 0 dává původní LJ ---------------------------------------------
    r = np.linspace(0.4e-9, 3e-9, 20)
    check("lj_plus_vdw_force s H = 0 == lj_force",
          bool(np.allclose(lj_plus_vdw_force(r, U0, Ra, 0.0, R), lj_force(r, U0, Ra))),
          "20 bodů")

    # --- 7. Δf(d) je z dálky přitažlivé a monotónní ---------------------------
    A = 0.9e-9
    force_fn = lambda r: lj_plus_vdw_force(r, U0, Ra, H, R)
    grid = np.linspace(3e-9, 30e-9, 60)
    vals = np.array([frequency_shift(d, A, K_CANT, F0, force_fn) for d in grid])
    check("Δf(d) < 0 a roste k nule s rostoucím d (3-30 nm)",
          bool(np.all(vals < 0) and np.all(np.diff(vals) > 0)),
          f"Δf(3 nm) = {vals[0]:.2f} Hz, Δf(30 nm) = {vals[-1]:.3f} Hz")

    print()
    if all(checks):
        print(f"Souhrn: {len(checks)}/{len(checks)} [OK]")
    else:
        print(f"Souhrn: {sum(checks)}/{len(checks)} [OK], "
              f"{len(checks) - sum(checks)} [SELHALO]")


if __name__ == "__main__":
    main()
