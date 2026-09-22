"""Korektnostní test režimu bez zpětné vazby a krokovacího callbacku.

Není to pytest test: skript vypíše [OK]/[SELHALO] pro každou kontrolu
a na konci souhrn (stejně jako test_mge.py/test_sum.py).

Kontroluje dvě nové věci ve stm_sim/sim.py, na kterých stojí interaktivní
nástroj:
- constant_height_scan(): sken s vypnutou smyčkou (režim konstantní výšky),
- callback v run_loop(): krokování výpočtu, které NESMÍ měnit výsledek.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import run_simulation as rs
from stm_sim.plant import FirstOrderPlant
from stm_sim.sim import constant_height_scan, run_loop
from stm_sim.surface import step_surface

X_EDGE = 3e-9
H = 0.2e-9


def _bez_sumu_sken(z_fixed, t_end, surface_fn):
    return constant_height_scan(
        surface_fn=surface_fn, v=rs.v, kappa=rs.kappa, V=rs.V, I_set=rs.I_set,
        g_set=rs.g_set, g_contact=rs.g_contact, dt=rs.dt, t_end=t_end,
        z_fixed=z_fixed,
    )


def main():
    checks = []

    def check(description, ok, detail):
        checks.append(ok)
        print(f"[{'OK' if ok else 'SELHALO'}] {description}   ({detail})")

    surface_fn = lambda x: step_surface(x, X_EDGE, H)
    t_end = (X_EDGE + 2e-9) / rs.v

    # --- 1. proud přesně kopíruje exp(-2*kappa*g), žádná integrace ---------
    z_fixed = rs.g_set + H + 0.3e-9   # vysoko nad schodem, náraz nehrozí
    res = _bez_sumu_sken(z_fixed, t_end, surface_fn)
    check("Sken v konstantní výšce doběhl bez nárazu (hrot vysoko nad schodem)",
          not res.crashed, f"z_fixed = {z_fixed * 1e9:.3f} nm, "
          f"min g = {min(res.g) * 1e9:.3f} nm")
    check("z_tip je po celý sken konstantní (regulátor opravdu nic nedělá)",
          all(z == z_fixed for z in res.z_tip),
          f"unikátních hodnot z_tip: {len(set(res.z_tip))}")

    # I(x) musí sedět na analytický vzorec c*V*exp(-2*kappa*g) s kalibrací
    # I(g_set) = I_set, tedy I = I_set * exp(-2*kappa*(g - g_set)).
    ocekavane = [rs.I_set * math.exp(-2 * rs.kappa * (g - rs.g_set)) for g in res.g]
    rel = max(abs(a - b) / b for a, b in zip(res.I, ocekavane))
    check("I(x) sedí na I_set*exp(-2*kappa*(g - g_set)) (rel. chyba < 1e-12)",
          rel < 1e-12, f"max rel. chyba = {rel:.3e}")

    # Poměr proudů před a za hranou musí být přesně exp(2*kappa*H).
    i_pred = res.I[0]
    i_za = res.I[-1]
    pomer_ocekavany = math.exp(2 * rs.kappa * H)
    check("Poměr I(za hranou)/I(před hranou) = exp(2*kappa*H)",
          abs(i_za / i_pred - pomer_ocekavany) / pomer_ocekavany < 1e-12,
          f"poměr = {i_za / i_pred:.4f}, očekáváno {pomer_ocekavany:.4f} "
          f"({i_pred * 1e12:.2f} pA -> {i_za * 1e12:.2f} pA)")

    # --- 2. bez regulátoru dojde na vyšší hraně k nárazu -------------------
    # z_fixed těsně nad povrchem: za hranou musí mezera spadnout pod g_contact.
    z_narazovy = H + rs.g_contact - 0.02e-9
    res_naraz = _bez_sumu_sken(z_narazovy, t_end, surface_fn)
    check("Bez zpětné vazby dojde na hraně k nárazu",
          res_naraz.crashed,
          f"z_fixed = {z_narazovy * 1e9:.3f} nm, náraz v "
          f"x = {res_naraz.x[-1] * 1e9:.3f} nm (hrana {X_EDGE * 1e9:.1f} nm)")

    # Stejná situace SE zpětnou vazbou musí dopadnout jinak (hrot uhne).
    controller = rs.build_controller()
    plant = FirstOrderPlant(z0=rs.g_set, T_sys=rs.T_SYS)
    res_smycka = run_loop(
        controller=controller, plant=plant, surface_fn=surface_fn,
        v=rs.v, kappa=rs.kappa, V=rs.V, I_set=rs.I_set, g_set=rs.g_set,
        g_contact=rs.g_contact, dt=rs.dt, t_end=t_end,
    )
    check("Se zapnutou zpětnou vazbou hrot přes stejnou hranu projede",
          not res_smycka.crashed,
          f"min g = {min(res_smycka.g) * 1e9:.3f} nm > g_contact = "
          f"{rs.g_contact * 1e9:.3f} nm")
    check("Se smyčkou se hrot nad hranou zvedne, bez ní ne",
          max(res_smycka.z_tip) - min(res_smycka.z_tip) > 0.5 * H,
          f"zdvih z_tip = {(max(res_smycka.z_tip) - min(res_smycka.z_tip)) * 1e9:.3f} nm, "
          f"H = {H * 1e9:.2f} nm")

    # --- 3. callback nemění výsledek a vidí ty správné veličiny ------------
    def bez_callbacku():
        c = rs.build_controller()
        p = FirstOrderPlant(z0=rs.g_set, T_sys=rs.T_SYS)
        return run_loop(controller=c, plant=p, surface_fn=surface_fn,
                        v=rs.v, kappa=rs.kappa, V=rs.V, I_set=rs.I_set,
                        g_set=rs.g_set, g_contact=rs.g_contact, dt=rs.dt,
                        t_end=t_end)

    zachyceno = []
    c2 = rs.build_controller()
    p2 = FirstOrderPlant(z0=rs.g_set, T_sys=rs.T_SYS)
    res_cb = run_loop(controller=c2, plant=p2, surface_fn=surface_fn,
                      v=rs.v, kappa=rs.kappa, V=rs.V, I_set=rs.I_set,
                      g_set=rs.g_set, g_contact=rs.g_contact, dt=rs.dt,
                      t_end=t_end, callback=lambda i, s: zachyceno.append((i, s)))
    res_ref = bez_callbacku()

    stejne = all(
        np.array_equal(np.asarray(getattr(res_cb, pole), dtype=np.float64),
                       np.asarray(getattr(res_ref, pole), dtype=np.float64))
        for pole in ("t", "x", "z_tip", "g", "I", "I_meas", "e")
    )
    check("callback nemění výsledek (bit-přesná shoda všech průběhů)",
          stejne and res_cb.crashed == res_ref.crashed,
          f"kroků: {len(res_ref.t)}")
    check("callback se zavolal jednou za krok",
          len(zachyceno) == len(res_ref.t),
          f"volání = {len(zachyceno)}, kroků = {len(res_ref.t)}")

    i_posl, stav = zachyceno[len(zachyceno) // 2]
    check("callback dostává veličiny odpovídajícího kroku",
          stav["e"] == res_ref.e[i_posl] and stav["g"] == res_ref.g[i_posl]
          and stav["x"] == res_ref.x[i_posl],
          f"krok {i_posl}: e = {stav['e']:.4e}, g = {stav['g'] * 1e9:.4f} nm")
    check("callback nese i mezikroky regulátoru (y) a novou polohu hrotu",
          stav["y"] is not None and stav["z_tip_novy"] == res_ref.z_tip[i_posl + 1],
          f"y = {stav['y']:.4e} m, z_tip_novy = {stav['z_tip_novy'] * 1e9:.4f} nm")

    print("\nVŠECHNY KONTROLY PROŠLY" if all(checks) else "\nNĚKTERÁ KONTROLA SELHALA")


if __name__ == "__main__":
    main()
