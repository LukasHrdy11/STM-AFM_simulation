"""Korektnostní test povrchu atomů (periodická Gaussovská mřížka,
stm_sim/surface.py::atoms_surface).

Není to pytest test: skript vypíše [OK]/[SELHALO] pro každou kontrolu
a na konci souhrn (stejně jako test_fig_5_11.py/test_mge.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import run_simulation as rs
import run_simulation_afm as rs_afm
from stm_sim.controller import PIController, gain_from_tau
from stm_sim.plant import FirstOrderPlant
from stm_sim.sim import run_loop
from stm_sim.surface import atoms_surface

X_START = 1e-9       # m
H = 0.15e-9          # m
SPACING = 0.6e-9      # m
SIGMA = 0.10e-9        # m


def main():
    checks = []

    def check(description, ok, detail):
        checks.append(ok)
        print(f"[{'OK' if ok else 'SELHALO'}] {description}   ({detail})")

    h = lambda x: atoms_surface(x, X_START, H, SPACING, SIGMA)

    # --- 1. h(x) ~ 0 daleko před X_START ------------------------------------
    h_pred = h(X_START - 5 * SPACING)
    check("h(x) ~ 0 daleko před X_START (< 1e-3*H)", abs(h_pred) < 1e-3 * H,
          f"h(x_start - 5*spacing) = {h_pred * 1e12:.6f} pm")

    # --- 2. h na středu atomu ~ H --------------------------------------------
    x_c0 = X_START + 0.5 * SPACING
    h_c0 = h(x_c0)
    check("h(x) na středu prvního atomu ~ H (rel. tolerance 1e-3)",
          abs(h_c0 - H) / H < 1e-3, f"h(x_c0) = {h_c0 * 1e9:.6f} nm, H = {H * 1e9:.4f} nm")

    # --- 3. periodicita: h na středech dalších atomů je stejné ---------------
    # Tolerance NENÍ strojová nula: první atom má oříznutého levého souseda
    # (i_c - 2 se ořezává na 0), takže mu chybí příspěvek nejbližšího
    # souseda řádu exp(-(spacing/sigma)^2/2) ~ 1,5e-8 relativně k H (viz
    # docstring atoms_surface) - vnitřní atomy tenhle příspěvek mají.
    stredy = [X_START + (i + 0.5) * SPACING for i in range(6)]
    hodnoty_stredu = [h(xc) for xc in stredy]
    check("Periodicita: h(x) na středech atomů je stejné pro několik period",
          max(hodnoty_stredu) - min(hodnoty_stredu) < 1e-6 * H,
          f"hodnoty = {[f'{v*1e9:.9f}' for v in hodnoty_stredu]} nm")

    # --- 4. oddělenost: v půlce mezi atomy je h výrazně menší než H ----------
    x_mid = X_START + SPACING  # půlka mezi prvním a druhým atomem
    h_mid = h(x_mid)
    check("Atomy jsou vizuálně oddělené (h v půlce mezi nimi < 0,05*H)",
          h_mid < 0.05 * H, f"h(mid) = {h_mid * 1e12:.4f} pm, 0,05*H = {0.05*H*1e12:.4f} pm")

    # --- 5. meze: h(x) v [0, H] na hustém gridu přes víc atomů ----------------
    xs = np.linspace(X_START, X_START + 6 * SPACING, 2000)
    vals = [h(x) for x in xs]
    check("h(x) zůstává v rozsahu [0, H] přes víc atomů",
          min(vals) >= 0.0 and max(vals) <= H * (1 + 1e-9),
          f"min={min(vals) * 1e9:.6f} nm, max={max(vals) * 1e9:.6f} nm")

    # --- 6. čistá funkce x: forward vs. reversed dotazy dají stejný výsledek -
    poradi_1 = [h(x) for x in xs]
    poradi_2 = [h(x) for x in xs[::-1]][::-1]
    check("h(x) dává stejný výsledek nezávisle na pořadí volání (čistá funkce)",
          poradi_1 == poradi_2, "forward vs. reversed dotazy na stejné x")

    # --- 7. integrace se smyčkou run_loop: regulátor sleduje povrch bez pádu -
    v = rs.v
    scan_len = X_START + 6 * SPACING + 1e-9
    t_end = scan_len / v
    K_I = gain_from_tau(rs.kappa, rs.tau)
    controller = PIController(y0=rs.g_set, K_I=K_I, K_P=rs.K_P)
    plant = FirstOrderPlant(z0=rs.g_set, T_sys=rs.T_SYS)
    result = run_loop(controller=controller, plant=plant, surface_fn=h,
                       v=v, kappa=rs.kappa, V=rs.V, I_set=rs.I_set,
                       g_set=rs.g_set, g_contact=rs.g_contact,
                       dt=rs.dt, t_end=t_end)
    check("run_loop nad atoms povrchem doběhne bez nárazu (rozumné H vůči g_set)",
          not result.crashed, f"crashed={result.crashed}")
    rms = float(np.sqrt(np.mean((np.array(result.g) - rs.g_set) ** 2)))
    check("Regulátor sleduje atoms povrch se sledovací chybou menší než H",
          rms < H, f"rms(g - g_set) = {rms * 1e12:.2f} pm, H = {H * 1e12:.0f} pm")

    # --- 8. SURFACE = "atoms" je zapojený v obou spouštěčích -----------------
    puvodni_surface_stm = rs.SURFACE
    rs.SURFACE = "atoms"
    surf_stm = rs.build_surface()
    check('run_simulation.build_surface() akceptuje SURFACE = "atoms"',
          callable(surf_stm) and surf_stm(rs.ATOMS_START - 1e-9) < 1e-3 * rs.ATOMS_H,
          f"surf(atoms_start - 1nm) = {surf_stm(rs.ATOMS_START - 1e-9):.3e} m")
    rs.SURFACE = puvodni_surface_stm

    puvodni_surface_afm = rs_afm.SURFACE
    rs_afm.SURFACE = "atoms"
    surf_afm = rs_afm.build_surface()
    check('run_simulation_afm.build_surface() akceptuje SURFACE = "atoms"',
          callable(surf_afm) and surf_afm(rs_afm.ATOMS_START - 1e-9) < 1e-3 * rs_afm.ATOMS_H,
          f"surf(atoms_start - 1nm) = {surf_afm(rs_afm.ATOMS_START - 1e-9):.3e} m")
    rs_afm.SURFACE = puvodni_surface_afm

    # --- 9. drive-by oprava: SURFACE = "step"/"ramp" v run_simulation.py
    #    nespadne (dřívější bug odkazoval na nedefinované x_edge/H) ----------
    for surf_test in ("step", "ramp"):
        rs.SURFACE = surf_test
        try:
            surf_fn = rs.build_surface()
            surf_fn(rs.x1_edge)
            ok = True
            detail = "beze změny"
        except NameError as exc:
            ok = False
            detail = str(exc)
        check(f'run_simulation.build_surface() pro SURFACE = "{surf_test}" '
              f"nespadne na NameError", ok, detail)
    rs.SURFACE = puvodni_surface_stm

    print("\nVŠECHNY KONTROLY PROŠLY" if all(checks) else "\nNĚKTERÁ KONTROLA SELHALA")


if __name__ == "__main__":
    main()
