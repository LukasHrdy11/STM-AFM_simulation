"""Korektnostní test povrchu MGE (Mackey-Glassova rovnice, stm_sim/surface.py).

Není to pytest test: skript vypíše [OK]/[SELHALO] pro každou kontrolu
a na konci souhrn (stejně jako test_fig_5_11.py/test_sum.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import run_simulation as rs
from stm_sim.controller import PIController, gain_from_tau
from stm_sim.plant import FirstOrderPlant
from stm_sim.sim import run_loop
from stm_sim.surface import _mackey_glass_series, make_mackey_glass_surface

X_EDGE = 1e-9   # m
H = 0.2e-9      # m
X_STEP = 0.05e-9  # m


def main():
    checks = []

    def check(description, ok, detail):
        checks.append(ok)
        print(f"[{'OK' if ok else 'SELHALO'}] {description}   ({detail})")

    # --- 1. syrová řada je deterministická a v [0, 1] ----------------------
    serie_a = _mackey_glass_series(2000)
    serie_b = _mackey_glass_series(2000)
    check("_mackey_glass_series je deterministická (stejný vstup -> stejný výstup)",
          serie_a == serie_b, "beze změny mezi dvěma voláními")
    check("_mackey_glass_series je normalizovaná do [0, 1]",
          min(serie_a) == 0.0 and max(serie_a) == 1.0,
          f"min={min(serie_a):.4f}, max={max(serie_a):.4f}")

    # --- 2. řada je chaotická, ne konstantní/periodická s krátkou periodou -
    std = float(np.std(serie_a))
    check("Rozptyl řady je netriviální (std > 0.1, řada nevymizí do konstanty)",
          std > 0.1, f"std = {std:.4f}")

    # --- 3. h(x) povrchu (jeden krok) ---------------------------------------
    h = make_mackey_glass_surface([(X_EDGE, H)], x_span_total=X_EDGE + 10e-9, x_step=X_STEP)
    check("h(x) == 0 před hranou (x < x_edge)", h(X_EDGE - 1e-9) == 0.0,
          f"h(x_edge - 1nm) = {h(X_EDGE - 1e-9):.3e} m")
    check("h(x_edge) == 0 na hraně", h(X_EDGE) == 0.0, f"h(x_edge) = {h(X_EDGE):.3e} m")

    xs = np.linspace(X_EDGE, X_EDGE + 10e-9, 500)
    vals = [h(x) for x in xs]
    check("h(x) za hranou zůstává v rozsahu [0, H]",
          min(vals) >= 0.0 and max(vals) <= H + 1e-18,
          f"min={min(vals) * 1e9:.4f} nm, max={max(vals) * 1e9:.4f} nm, H={H * 1e9:.4f} nm")
    check("h(x) je nekonstantní za hranou (profil se opravdu mění)",
          max(vals) - min(vals) > 0.3 * H,
          f"rozkmit = {(max(vals) - min(vals)) * 1e9:.4f} nm")

    # lineární interpolace mezi vzorky: hodnota na vzorkovacím bodě musí
    # sedět a hodnota v půlce kroku musí ležet mezi sousedními vzorky
    x_sample = X_EDGE + 5 * X_STEP
    h_sample = h(x_sample)
    h_next = h(x_sample + X_STEP)
    h_mid = h(x_sample + X_STEP / 2)
    lo, hi = sorted([h_sample, h_next])
    check("Lineární interpolace mezi vzorky (hodnota v půlce kroku leží mezi sousedy)",
          lo - 1e-15 <= h_mid <= hi + 1e-15,
          f"h_mid={h_mid * 1e9:.5f} nm, sousedi=[{lo * 1e9:.5f}, {hi * 1e9:.5f}] nm")

    # --- 4. h(x) je čistá funkce x (idempotentní, nezávislá na pořadí volání) -
    poradi_1 = [h(x) for x in xs]
    poradi_2 = [h(x) for x in xs[::-1]][::-1]
    check("h(x) dává stejný výsledek nezávisle na pořadí volání (čistá funkce)",
          poradi_1 == poradi_2, "forward vs. reversed dotazy na stejné x")

    # --- 5. konzistence s x_span: klamp za koncem tabulky -------------------
    h_far_1 = h(X_EDGE + 50e-9)
    h_far_2 = h(X_EDGE + 100e-9)
    check("Dotaz daleko za x_span vrátí konstantní (clampovanou) hodnotu, ne pád",
          h_far_1 == h_far_2, f"h(+50nm) = {h_far_1 * 1e9:.4f} nm, h(+100nm) = {h_far_2 * 1e9:.4f} nm")

    # --- 6. integrace se smyčkou run_loop: regulátor sleduje povrch bez pádu -
    v = rs.v
    scan_len = 20e-9
    t_end = scan_len / v
    surf = make_mackey_glass_surface([(1e-9, H)], x_span_total=scan_len, x_step=X_STEP)
    K_I = gain_from_tau(rs.kappa, rs.tau)
    controller = PIController(y0=rs.g_set, K_I=K_I, K_P=rs.K_P)
    plant = FirstOrderPlant(z0=rs.g_set, T_sys=rs.T_SYS)
    result = run_loop(controller=controller, plant=plant, surface_fn=surf,
                       v=v, kappa=rs.kappa, V=rs.V, I_set=rs.I_set,
                       g_set=rs.g_set, g_contact=rs.g_contact,
                       dt=rs.dt, t_end=t_end)
    check("run_loop nad MGE povrchem doběhne bez nárazu (rozumné H vůči g_set)",
          not result.crashed, f"crashed={result.crashed}")
    rms = float(np.sqrt(np.mean((np.array(result.g) - rs.g_set) ** 2)))
    check("Regulátor sleduje MGE povrch se sledovací chybou menší než H",
          rms < H, f"rms(g - g_set) = {rms * 1e12:.2f} pm, H = {H * 1e12:.0f} pm")

    # --- 7. SURFACE = "MGE" je zapojený v run_simulation.py ----------------
    rs.SURFACE = "MGE"
    surf_rs = rs.build_surface()
    check('run_simulation.build_surface() akceptuje SURFACE = "MGE"',
          callable(surf_rs) and surf_rs(rs.x1_edge - 1e-9) == 0.0,
          f"surf(x_edge - 1nm) = {surf_rs(rs.x1_edge - 1e-9):.3e} m")
    rs.SURFACE = "step"  # vrátit výchozí stav modulu

    # --- 8. více nezávislých MGE hran za sebou ------------------------------
    steps = [(2e-9, 0.2e-9), (7e-9, 0.15e-9), (12e-9, 0.25e-9)]
    h_multi = make_mackey_glass_surface(steps, x_span_total=17e-9, x_step=X_STEP)

    check("h(x) == 0 před první hranou (více kroků)", h_multi(1e-9) == 0.0,
          f"h(1nm) = {h_multi(1e-9):.3e} m")

    for x_edge_i, H_i in steps:
        xs_seg = np.linspace(x_edge_i + X_STEP, x_edge_i + 4e-9, 200)
        vals_seg = [h_multi(x) for x in xs_seg]
        check(f"Úsek za hranou {x_edge_i * 1e9:.0f} nm zůstává v [0, H={H_i * 1e9:.2f} nm]",
              min(vals_seg) >= 0.0 and max(vals_seg) <= H_i + 1e-18,
              f"min={min(vals_seg) * 1e9:.4f} nm, max={max(vals_seg) * 1e9:.4f} nm")

    # úseky jsou nezávislé: normalizované (bezrozměrné, /H) profily dvou
    # úseků nejsou (skoro) shodné, i když by při sdílené historii/vstupu
    # vypadaly stejně (jen jinak škálované amplitudou)
    n_common = 60
    profil_1 = [h_multi(steps[0][0] + X_STEP + i * X_STEP) / steps[0][1] for i in range(n_common)]
    profil_2 = [h_multi(steps[1][0] + X_STEP + i * X_STEP) / steps[1][1] for i in range(n_common)]
    rozdil = float(np.mean(np.abs(np.array(profil_1) - np.array(profil_2))))
    check("Nezávislé úseky nemají shodný (jen amplitudou přeškálovaný) profil",
          rozdil > 0.05, f"průměrný rozdíl normalizovaných profilů = {rozdil:.4f}")

    # vnitřní hrana nemusí (na rozdíl od té úplně první) startovat od 0 -
    # nová MG sekvence začíná svou vlastní normalizovanou hodnotou, jen musí
    # zůstat v [0, H] daného úseku
    h_hrana2 = h_multi(steps[1][0])
    check("h(x) na vnitřní hraně leží v rozsahu [0, H] daného úseku",
          0.0 <= h_hrana2 <= steps[1][1] + 1e-18,
          f"h(hrana 2) = {h_hrana2 * 1e9:.4f} nm, H úseku = {steps[1][1] * 1e9:.2f} nm")

    print("\nVŠECHNY KONTROLY PROŠLY" if all(checks) else "\nNĚKTERÁ KONTROLA SELHALA")


if __name__ == "__main__":
    main()
