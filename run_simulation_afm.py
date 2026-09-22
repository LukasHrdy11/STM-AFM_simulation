"""Jednotný spouštěč simulace zpětné vazby FM-AFM (qPlus).

Mirror run_simulation.py (viz tam pro STM). Rozdíly:
- GAIN_MODE: "fixed" (citlivost kappa_eff se spočte jednou v d_set, jako
  u STM) vs. "local" (přepočítává se v každém kroku podle aktuální
  polohy d - demonstruje nestabilitu, když se hrot přiblíží k minimu
  Δf(d), viz afm_sim/sim.py).
- Fyzikální hodnoty (k_cant, f0, A, U0, Ra, tau) jsou NEMĚŘENÉ,
  ilustrativní odhady - na rozdíl od STM kappa se kalibrovat NEPOVEDLO:
  `export/kalibrace_afm/souhrn.md` ukazuje, že atomární LJ model neumí
  vysvětlit, jak pomalu naměřené Δf(z) klesá (viz tam, sekce Meze) - U0/Ra
  proto zůstávají beze změny.
- Šum měřeného Δf a šum amplitudy jsou implementovány (`afm_sim/noise.py`, kalibrováno v
  `export/sum_frekvence_afm/souhrn.md`), zapojení viz SUM_FREKVENCE / SUM_AMPLITUDY níže.
  Šum mezery (`gap_noise`) zůstává nezapojený - nezměřený, viz tamní Meze.

Jak simulaci upravit - stačí změnit hodnoty v sekci NASTAVENÍ níže.
Graf se neukazuje živě, jen se uloží do export/run_afm/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from afm_sim import noise
from afm_sim.plotting import (
    fig_frekvence_kolem_udalosti, fig_prubehy,
)
from afm_sim.frequency_shift import frequency_shift, gain_from_tau, kappa_eff
from afm_sim.sim import run_loop
from afm_sim.tip_force import lj_force, r_min
from afm_sim.tip_force_atoms import AtomRow
from stm_sim.controller import (
    IntegralController, PIController, ProportionalController,
)
from stm_sim.plant import FirstOrderPlant
from stm_sim.surface import atoms_surface, ramp_surface, step_surface

EXPORT_DIR = Path(__file__).parent / "export" / "run_afm"

# ============================== NASTAVENÍ ================================

CONTROLLER = "PI"    # "P" | "I" | "PI"
SURFACE = "atoms"    # "step" | "ramp" | "atoms" (reuse stm_sim.surface;
                     # MGE se sem nezapojuje) | "atoms-AFM" (jen AFM: řada
                     # BODOVÝCH atomů, síla se sčítá přes atomy ve 3D, viz
                     # afm_sim/tip_force_atoms.py a ATOMS_AFM_* níže)
GAIN_MODE = "fixed"  # "fixed" | "local" - přepnutím se STEJNÝMI d_set/H
                     # níže jde vidět rozdíl: "fixed" projede dip u hrany
                     # beze změny, "local" nahlásí NESTABILITA, jakmile
                     # se hrot přiblíží k minimu Δf(d) a citlivost změní
                     # znaménko. Na SURFACE = "atoms" to "local" spolehlivě
                     # ukáže hned na prvním atomu - to je očekávané, ne bug.
T_SYS = 100e-6       # setrvačnost akčního členu [s]
BACKWARD_SCAN = True

# --- Šum (viz afm_sim/noise.py) ---
SUM_FREKVENCE = True  # šum na měřeném Δf (bílý za propustí, viz souhrn.md)
SUM_AMPLITUDY = True  # šum amplitudy oscilace (std ~1,9 pm); do Δf vstupuje přes Δf(A). Losuje
                      # se ze SAMOSTATNÉHO generátoru, takže jeho zapnutí nemění šum Δf.
SEED = 42              # seed generátoru; jeden rng pro forward i backward

# --- Fyzikální parametry cantileveru a LJ potenciálu (NEMĚŘENÉ odhady) ---
# Pokus o kalibraci U0/Ra proti reálné Δf(z) (export/kalibrace_afm/souhrn.md)
# NEUSPĚL - atomární LJ model s libovolným U0 neumí vysvětlit, jak pomalu
# naměřené Δf klesá (poměr délky poklesu ~50x). Hodnoty proto zůstávají
# beze změny, viz tamní sekce Meze.
k_cant = 1800.0       # N/m, typická tuhost qPlus (tab. 19.1)
f0 = 30e3             # Hz
A = 0.3e-9            # m, amplituda oscilace
U0 = 0.3 * 1.602e-19  # J (~0.3 eV), hloubka LJ jámy - odhad
Ra = 0.3e-9           # m, LJ nulový bod - odhad

d_set = 1.0e-9        # m, pracovní bod (musí ležet v přitažlivé větvi
                      # PŘED minimem Δf(d), kontrolováno assertem níže).
                      # Zvednuto z 0,72e-9 na 1,0e-9, aby rezerva do
                      # d_contact (363 pm) bezpečně pojala ATOMS_H = 0,15 nm
                      # (stejná výška atomu jako v run_simulation.py - jinak
                      # by povrch v porovnání STM/AFM nebyl stejný).
d_contact = r_min(Ra) + A  # m; dolní úvrať oscilace (d_contact - A) musí
                           # zůstat nad r_min, jinak hrot narazí uvnitř
                           # oscilačního cyklu, ne jen v mean poloze

tau = 1e-3            # s (časová konstanta I-složky regulátoru)
K_P = 5e-11           # m; ilustrativní - není kalibrováno, srov. STM K_P

v = 65e-9             # m/s
x_edge = 3e-9         # m
H = 0.08e-9           # m (výška schodu)
w = 0.3e-9            # m (šířka hrany; jen pro SURFACE = "ramp")
ATOMS_START = 3e-9        # m (odkud mřížka atomů začíná; jen pro SURFACE = "atoms")
ATOMS_H = 0.15e-9          # m (výška atomu; stejná hodnota jako v run_simulation.py)
ATOMS_SPACING = 0.6e-9     # m (rozestup mezi atomy)
ATOMS_SIGMA = 0.10e-9      # m (šířka atomu)
ATOMS_POCET = 6            # počet atomů, přes které se skenuje (řídí jen t_end)

# --- Povrch "atoms-AFM" (jen SURFACE = "atoms-AFM"): řada bodových atomů ---
# Povrch nemá výšku h(x): Δf vzniká součtem sil od všech atomů (svislá složka LJ síly
# ve 3D vzdálenosti), takže hrot atom "cítí" i mimo pozici přímo nad ním. Boční
# rozlišení vychází z dosahu síly, ne z parametru (viz analysis/analyza_atoms_afm.py).
ATOMS_AFM_START = 3e-9       # m (odkud řada začíná; první atom je v START + SPACING/2)
ATOMS_AFM_SPACING = 0.6e-9   # m (rozestup atomů)
ATOMS_AFM_POCET = 6          # počet atomů v řadě
ATOMS_AFM_Y = 0.0            # m (posun skenovací linky od řady; 0 = přímo nad atomy)
ATOMS_AFM_D_SET = 0.80e-9    # m (pracovní bod NAD atomem; d_set = 1,0 nm z bloku výše je pro
                             # bodové atomy příliš daleko - rozdíl Δf nad atomem a mezi atomy je
                             # tam ~0,02 Hz, pod šumem 0,08 Hz. Musí ležet nad d_contact s rezervou
                             # na "údolí" mezi atomy, kde hrot klesne, aby udržel Δf.)

dt = tau / 100
if SURFACE == "atoms":
    t_end = 20 * tau + (ATOMS_START + (ATOMS_POCET + 0.5) * ATOMS_SPACING) / v
elif SURFACE == "atoms-AFM":
    d_set = ATOMS_AFM_D_SET
    # Sken projede JEN řadu atomů: daleko od bodových atomů je síla nulová, regulátor by
    # hrot spustil dolů až na náraz (na rozdíl od spojité mřížky `atoms` tu není podložka).
    # Proto začíná na začátku řady (X0_SKEN) a končí za posledním atomem, bez přidaných 20*tau.
    t_end = ATOMS_AFM_POCET * ATOMS_AFM_SPACING / v
else:
    t_end = 20 * tau + x_edge / v

X0_SKEN = ATOMS_AFM_START if SURFACE == "atoms-AFM" else 0.0   # m, počáteční x forward průjezdu

# ===========================================================================

force_fn = lambda r: lj_force(r, U0, Ra)

# Síla pro "atoms-AFM": závisí i na x. `force_fn_xz` jde do run_loop; `force_ref` je
# síla NAD prostředním atomem (nejhorší případ = nejblíž k minimu Δf(d), bez okrajových
# atomů, kterým chybí soused) - používá se pro setpoint, citlivost a kontrolu d_set.
# Kritérium nárazu `d <= d_contact` je svislé (z_tip nad rovinou atomů); mezi atomy je
# skutečná vzdálenost k nejbližšímu atomu větší, takže je konzervativní, ne chybné.
if SURFACE == "atoms-AFM":
    atoms_row = AtomRow(ATOMS_AFM_START, ATOMS_AFM_SPACING, ATOMS_AFM_POCET, y_offset=ATOMS_AFM_Y)
    x_ref = float(atoms_row.x[ATOMS_AFM_POCET // 2])
    force_fn_xz = lambda x, z: atoms_row.force_z(x, z, U0, Ra)
    force_ref = lambda z: atoms_row.force_z(x_ref, z, U0, Ra)
else:
    atoms_row = None
    force_fn_xz = None
    force_ref = force_fn


def _d_argmin_frequency_shift():
    """Najde polohu minima Δf(d) hledáním na mřížce mezi d_contact a 5*Ra.

    Použito pro sanity check v build_controller() - pracovní bod d_set
    musí ležet za tímto minimem (dál od vzorku), jinak jsme na
    nemonotónní/nefyzikální části Δf(d) (viz afm_sim/frequency_shift.py).
    """
    grid = np.linspace(d_contact, 5.0 * Ra, 400)
    vals = [frequency_shift(d, A, k_cant, f0, force_ref) for d in grid]
    return grid[int(np.argmin(vals))]


def build_controller():
    d_argmin = _d_argmin_frequency_shift()
    if not (d_set > d_argmin):
        raise AssertionError(
            f"d_set = {d_set:.3e} m musí ležet za minimem Δf(d) "
            f"(d_argmin = {d_argmin:.3e} m), jinak je citlivost "
            f"kappa_eff v pracovním bodě nedefinovaná/záporná."
        )
    k_eff = kappa_eff(d_set, A, k_cant, f0, force_ref)
    K_I = gain_from_tau(k_eff, tau)
    if CONTROLLER == "P":
        return ProportionalController(base=d_set, K_P=K_P)
    if CONTROLLER == "I":
        return IntegralController(y0=d_set, K_I=K_I)
    if CONTROLLER == "PI":
        return PIController(y0=d_set, K_I=K_I, K_P=K_P)
    raise ValueError(f"Neznámý CONTROLLER: {CONTROLLER!r} (očekávám P/I/PI)")


def build_surface():
    if SURFACE == "step":
        return lambda x: step_surface(x, x_edge, H)
    if SURFACE == "ramp":
        return lambda x: ramp_surface(x, x_edge, H, w)
    if SURFACE == "atoms":
        return lambda x: atoms_surface(x, ATOMS_START, ATOMS_H, ATOMS_SPACING, ATOMS_SIGMA)
    if SURFACE == "atoms-AFM":
        return lambda x: 0.0    # referenční rovina = rovina atomů; výšku nese force_fn_xz
    raise ValueError(f"Neznámý SURFACE: {SURFACE!r} (očekávám step/ramp/atoms/atoms-AFM)")


def build_noise():
    """Vrátí (freq_noise, gap_noise) - jedny instance pro oba průjezdy."""
    rng = np.random.default_rng(SEED)
    freq_noise = noise.sum_frekvence(rng) if SUM_FREKVENCE else None
    gap_noise = None  # nezměřeno pro AFM, viz export/sum_frekvence_afm/souhrn.md
    return freq_noise, gap_noise


def build_amp_noise():
    """Šum amplitudy: samostatný generátor (SEED, 1), jedna instance pro oba průjezdy."""
    if not SUM_AMPLITUDY:
        return None
    return noise.sum_amplitudy(np.random.default_rng([SEED, 1]))


def main():
    df_set = frequency_shift(d_set, A, k_cant, f0, force_ref)
    controller = build_controller()
    plant = FirstOrderPlant(z0=d_set, T_sys=T_SYS)
    surface_fn = build_surface()
    adaptive_tau = tau if GAIN_MODE == "local" else None
    freq_noise, gap_noise = build_noise()
    amp_noise = build_amp_noise()

    result_fwd = run_loop(
        controller=controller, plant=plant, surface_fn=surface_fn,
        v=v, force_fn=force_fn, A=A, k_cant=k_cant, f0=f0,
        df_set=df_set, d_contact=d_contact, dt=dt, t_end=t_end,
        adaptive_tau=adaptive_tau, freq_noise=freq_noise, gap_noise=gap_noise,
        force_fn_xz=force_fn_xz, x0=X0_SKEN, amp_noise=amp_noise,
    )

    def hlaseni(result, label):
        if result.crashed:
            print(f"NÁRAZ ({label}) v t = {result.t[-1]:.3e} s, "
                  f"x = {result.x[-1]:.3e} m")
        elif result.unstable:
            print(f"NESTABILITA ({label}) v t = {result.t[-1]:.3e} s "
                  f"(hrot za minimem Δf, kappa_eff <= 0)")
        else:
            print(f"{label}: bez nárazu, bez ztráty stability.")
        d_min = min(result.d)
        print(f"d_min ({label}) = {d_min * 1e9:.4f} nm (d_set = {d_set * 1e9:.4f} nm)")

    hlaseni(result_fwd, "forward")

    result_bwd = None
    if BACKWARD_SCAN and not result_fwd.crashed and not result_fwd.unstable:
        result_bwd = run_loop(
            controller=controller, plant=plant, surface_fn=surface_fn,
            v=-v, force_fn=force_fn, A=A, k_cant=k_cant, f0=f0,
            df_set=df_set, d_contact=d_contact, dt=dt, t_end=t_end,
            x0=result_fwd.x[-1], t0=result_fwd.t[-1] + dt,
            adaptive_tau=adaptive_tau, freq_noise=freq_noise, gap_noise=gap_noise,
            force_fn_xz=force_fn_xz, amp_noise=amp_noise,
        )
        hlaseni(result_bwd, "backward")

    fig = fig_prubehy(
        result_fwd=result_fwd, surface_fn=surface_fn, d_contact=d_contact,
        df_set=df_set, result_bwd=result_bwd, atoms_row=atoms_row,
        titulek=f"{CONTROLLER}, {SURFACE}, GAIN_MODE = {GAIN_MODE}, "
                f"T_sys = {T_SYS * 1e6:.0f} µs",
    )

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_bwd" if (BACKWARD_SCAN and result_bwd is not None) else ""
    out_path = EXPORT_DIR / f"{CONTROLLER.lower()}_{SURFACE}{suffix}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Graf uložen do {out_path}")

    _uloz_graf_frekvence_kolem_udalosti(result_fwd, result_bwd, suffix)


def _uloz_graf_frekvence_kolem_udalosti(result_fwd, result_bwd, suffix,
                                         n_okno=300):
    """Uloží zoom rezonanční frekvence kolem nárazu/ztráty stability.

    Samotné kreslení je v afm_sim/plotting.py (sdílené s interaktivním
    nástrojem); tady se jen vybere, který průjezd událost obsahuje, a
    výsledek se uloží na disk.
    """
    if result_fwd.crashed or result_fwd.unstable:
        result, label = result_fwd, "forward"
    elif result_bwd is not None and (result_bwd.crashed or result_bwd.unstable):
        result, label = result_bwd, "backward"
    else:
        print("Bez nárazu/nestability v tomto běhu - graf f(t) kolem "
              "události se nevytváří.")
        return

    fig = fig_frekvence_kolem_udalosti(result, f0, label, n_okno=n_okno)

    out_path = EXPORT_DIR / f"{CONTROLLER.lower()}_{SURFACE}{suffix}_udalost.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Graf f(t) kolem události uložen do {out_path}")


if __name__ == "__main__":
    main()
