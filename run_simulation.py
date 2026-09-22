"""Jednotný spouštěč simulace zpětné vazby STM.

Jak simulaci upravit - stačí změnit hodnoty v sekci NASTAVENÍ níže:
- CONTROLLER: který regulátor použít ("P", "I", nebo "PI")
- SURFACE: jaký povrch skenovat ("step" = ostrý schod, "ramp" = schod
  s konečnou šířkou hrany w)
- T_SYS: setrvačnost akčního členu / piezo (0 = okamžitá odezva, jako v1)
- SUM_PROUDU, SUM_MEZERY: šum proudu a šum mezery (výchozí vypnuto),
  čísla viz stm_sim/noise.py a export/noise_proud/souhrn.md
Zbytek jsou fyzikální parametry smyčky (viz README).

Graf se neukazuje živě, jen se uloží do export/run/.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stm_sim import noise
from stm_sim.plotting import fig_prubehy
from stm_sim.controller import (
    IntegralController, PIController, ProportionalController, gain_from_tau,
)
from stm_sim.plant import FirstOrderPlant
from stm_sim.sim import run_loop
from stm_sim.surface import (
    atoms_surface, generate_mge_steps, make_mackey_glass_surface,
    ramp_surface, step_surface,
)

EXPORT_DIR = Path(__file__).parent / "export" / "run"

# ============================== NASTAVENÍ ================================

CONTROLLER = "PI"    # "P" | "I" | "PI"
SURFACE = "atoms"    # "step" | "ramp" | "MGE" (chaotický profil, Mackey-Glass)
                     # | "atoms" (periodická mřížka Gaussových atomů)
T_SYS = 100e-6       # setrvačnost akčního členu [s], 0 = okamžitá odezva
                     # (P-složka má smysl jen při T_SYS > 0, viz README)
BACKWARD_SCAN = True # po forward průjezdu jet stejnou linkou i zpět

# --- Šum (viz stm_sim/noise.py) ---
SUM_PROUDU = True        # šum elektroniky na měřeném I (bílý 7 kHz)
SUM_PROUDU_CARY = False  # k němu čáry 312 a 623 Hz (jen když SUM_PROUDU)
SUM_MEZERY = "stojici"   # None | "stojici" (horní odhad, stojící hrot)
                         # | "skeny" (nejhorší varianta, medián skenů)
SEED = 42                # seed generátoru; jeden rng pro forward i backward

# --- Fyzikální parametry (viz README) ---
I_set = 300e-12          # A
kappa = 6.8e9            # 1/m; změřeno z I(Z), medián 6 fitů (soubory x fwd/bwd),
                         # viz export/zavislost_proudu_na_z/souhrn.md
V = 2.5                  # V
tau = 200e-6             # s   (časová konstanta I-složky regulátoru)
K_P = 0.011e-9           # m; z hlaviček .sxm (K_p 5,6e-2 m/A * I_set), rozhodnutí autora;
                         # dřívější volba 4,5 nm (90 % meze stability 4,96 nm při
                         # T_SYS = 100 us, mez 1/(2*kappa) = 0,05 nm platí jen pro T_SYS = 0)
v = 65e-9                # m/s
g_set = 0.6e-9           # m
g_contact = 0.25e-9      # m
x1_edge = 3e-9            # m
H1 = 0.2e-9               # m (výška schodu)
w = 0.3e-9               # m (šířka hrany; jen pro SURFACE = "ramp")
MGE_POCET_SCHODU = 7       # počet schodů MGE povrchu (jen pro SURFACE = "MGE")
MGE_ROZESTUP = 0.6e-9      # m (rozestup mezi hranami; jen pro SURFACE = "MGE")
MGE_STEPS = generate_mge_steps(MGE_POCET_SCHODU, x1_edge, MGE_ROZESTUP, H1)
MGE_X_STEP = 0.05e-9      # m (vzorkovací krok profilu; jen pro SURFACE = "MGE")
ATOMS_START = 3e-9        # m (odkud mřížka atomů začíná; jen pro SURFACE = "atoms")
ATOMS_H = 0.15e-9         # m (výška atomu)
ATOMS_SPACING = 0.6e-9    # m (rozestup mezi atomy)
ATOMS_SIGMA = 0.10e-9     # m (šířka atomu, spacing/sigma = 6 - atomy vizuálně oddělené)
ATOMS_POCET = 6           # počet atomů, přes které se skenuje (řídí jen t_end, ne povrch)
dt = tau / 100
if SURFACE == "MGE":
    t_end = 20 * tau + (MGE_STEPS[-1][0] + 1e-9) / v  # najet přes celý profil + rezerva
elif SURFACE == "atoms":
    t_end = 20 * tau + (ATOMS_START + (ATOMS_POCET + 0.5) * ATOMS_SPACING) / v
else:
    t_end = 20 * tau + x1_edge / v

# ===========================================================================


def build_controller():
    K_I = gain_from_tau(kappa, tau)
    if CONTROLLER == "P":
        return ProportionalController(base=g_set, K_P=K_P)
    if CONTROLLER == "I":
        return IntegralController(y0=g_set, K_I=K_I)
    if CONTROLLER == "PI":
        return PIController(y0=g_set, K_I=K_I, K_P=K_P)
    raise ValueError(f"Neznámý CONTROLLER: {CONTROLLER!r} (očekávám P/I/PI)")


def build_surface():
    if SURFACE == "step":
        return lambda x: step_surface(x, x1_edge, H1)
    if SURFACE == "ramp":
        return lambda x: ramp_surface(x, x1_edge, H1, w)
    if SURFACE == "MGE":
        x_span_total = max(abs(v) * t_end, MGE_STEPS[-1][0] + MGE_X_STEP * 2)
        return make_mackey_glass_surface(MGE_STEPS, x_span_total, MGE_X_STEP)
    if SURFACE == "atoms":
        return lambda x: atoms_surface(x, ATOMS_START, ATOMS_H, ATOMS_SPACING, ATOMS_SIGMA)
    raise ValueError(f"Neznámý SURFACE: {SURFACE!r} (očekávám step/ramp/MGE/atoms)")


def build_noise():
    """Vrátí (current_noise, gap_noise) - jedny instance pro oba průjezdy."""
    rng = np.random.default_rng(SEED)
    current_noise = noise.sum_proudu(rng, cary=SUM_PROUDU_CARY) if SUM_PROUDU else None
    if SUM_MEZERY is None:
        gap_noise = None
    elif SUM_MEZERY == "stojici":
        gap_noise = noise.sum_mezery_stojici(rng)
    elif SUM_MEZERY == "skeny":
        gap_noise = noise.sum_mezery_skeny(rng)
    else:
        raise ValueError(f"Neznámý SUM_MEZERY: {SUM_MEZERY!r} (očekávám None/stojici/skeny)")
    return current_noise, gap_noise


def rms_odchylky(g):
    """rms odchylky mezery od g_set [m]."""
    return math.sqrt(sum((gi - g_set) ** 2 for gi in g) / len(g))


def main():
    controller = build_controller()
    plant = FirstOrderPlant(z0=g_set, T_sys=T_SYS)
    surface_fn = build_surface()
    current_noise, gap_noise = build_noise()

    result_fwd = run_loop(
        controller=controller, plant=plant, surface_fn=surface_fn,
        v=v, kappa=kappa, V=V, I_set=I_set, g_set=g_set,
        g_contact=g_contact, dt=dt, t_end=t_end,
        current_noise=current_noise, gap_noise=gap_noise,
    )

    if result_fwd.crashed:
        print(f"NÁRAZ (forward) v t = {result_fwd.t[-1]:.3e} s, "
              f"x = {result_fwd.x[-1]:.3e} m")
    else:
        print("Forward: bez nárazu.")
    g_min = min(result_fwd.g)
    print(f"g_min (forward) = {g_min * 1e9:.4f} nm  (g_set = {g_set * 1e9:.4f} nm)")
    print(f"rms(g - g_set) (forward) = {rms_odchylky(result_fwd.g) * 1e12:.2f} pm")

    result_bwd = None
    if BACKWARD_SCAN and not result_fwd.crashed:
        result_bwd = run_loop(
            controller=controller, plant=plant, surface_fn=surface_fn,
            v=-v, kappa=kappa, V=V, I_set=I_set, g_set=g_set,
            g_contact=g_contact, dt=dt, t_end=t_end,
            x0=result_fwd.x[-1], t0=result_fwd.t[-1] + dt,
            current_noise=current_noise, gap_noise=gap_noise,
        )
        if result_bwd.crashed:
            print(f"NÁRAZ (backward) v t = {result_bwd.t[-1]:.3e} s, "
                  f"x = {result_bwd.x[-1]:.3e} m")
        else:
            print("Backward: bez nárazu.")
        g_min_bwd = min(result_bwd.g)
        print(f"g_min (backward) = {g_min_bwd * 1e9:.4f} nm")
        print(f"rms(g - g_set) (backward) = {rms_odchylky(result_bwd.g) * 1e12:.2f} pm")

    fig = fig_prubehy(
        result_fwd=result_fwd, surface_fn=surface_fn, g_contact=g_contact,
        result_bwd=result_bwd,
        titulek=f"{CONTROLLER}, {SURFACE}, T_sys = {T_SYS * 1e6:.0f} µs",
    )

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_bwd" if BACKWARD_SCAN else ""
    if SUM_PROUDU or SUM_MEZERY is not None:
        suffix += "_sum"
    out_path = EXPORT_DIR / f"{CONTROLLER.lower()}_{SURFACE}{suffix}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Graf uložen do {out_path}")


if __name__ == "__main__":
    main()
