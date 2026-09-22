"""Korektnostní test povrchu atoms-AFM (`afm_sim/tip_force_atoms.py`).

Není to pytest test: skript vypíše [OK]/[SELHALO] pro každou kontrolu
a na konci souhrn (stejně jako test_vdw_sphere.py/test_atoms.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from afm_sim.frequency_shift import frequency_shift, gain_from_tau, kappa_eff
from afm_sim.noise import SIGMA_DF
from afm_sim.sim import run_loop
from afm_sim.tip_force import lj_force, r_min
from afm_sim.tip_force_atoms import AtomRow
from stm_sim.controller import PIController
from stm_sim.plant import FirstOrderPlant

U0 = 0.3 * 1.602e-19
Ra = 0.3e-9
K_CANT, F0, A = 1800.0, 30e3, 0.3e-9
SPACING = 0.6e-9


def df_atoms(row, x, d):
    return frequency_shift(d, A, K_CANT, F0, lambda z: row.force_z(x, z, U0, Ra))


def main():
    checks = []

    def check(description, ok, detail):
        checks.append(ok)
        print(f"[{'OK' if ok else 'SELHALO'}] {description}   ({detail})")

    # --- 1. jeden atom přímo pod hrotem = lj_force -----------------------------
    one = AtomRow(0.0, 1e-9, 1)
    x0 = float(one.x[0])
    z = np.array([0.5e-9, 0.8e-9, 1.4e-9])
    check("jeden atom přímo pod hrotem: F_z = lj_force(z) (dosavadní model)",
          bool(np.allclose(one.force_z(x0, z, U0, Ra), lj_force(z, U0, Ra), rtol=1e-12)),
          "shoda na 3 výškách")
    d = 0.9e-9
    df_ref = frequency_shift(d, A, K_CANT, F0, lambda r: lj_force(r, U0, Ra))
    check("jeden atom pod hrotem: Δf shodné s frequency_shift(lj_force)",
          abs(df_atoms(one, x0, d) / df_ref - 1.0) < 1e-12, f"Δf = {df_ref:.4f} Hz")

    # --- 2. průmět na osu z ------------------------------------------------------
    dx, zz = 0.4e-9, 0.7e-9
    r = np.hypot(dx, zz)
    f_ocekavane = lj_force(r, U0, Ra) * zz / r
    f_kod = one.force_z(x0 + dx, zz, U0, Ra)
    check("boční posun: F_z = lj_force(r) * z/r (průmět na svislou osu, ne velikost síly)",
          abs(f_kod / f_ocekavane - 1.0) < 1e-12 and abs(f_kod) < abs(lj_force(r, U0, Ra)),
          f"F_z = {f_kod:.3e} N, |F(r)| = {abs(lj_force(r, U0, Ra)):.3e} N")

    # --- 3. souměrnost a periodicita ---------------------------------------------
    row = AtomRow(0.0, SPACING, 15)
    xc = float(row.x[7])
    dd = 0.85e-9
    dl, dp = df_atoms(row, xc - 0.17e-9, dd), df_atoms(row, xc + 0.17e-9, dd)
    check("Δf je souměrné podle středu atomu (vnitřní atom, ±0,17 nm)",
          abs(dl / dp - 1.0) < 1e-9, f"{dl:.6f} vs. {dp:.6f} Hz")
    stredy = [df_atoms(row, float(row.x[i]), dd) for i in (6, 7, 8)]
    check("periodicita: Δf nad sousedními vnitřními atomy je stejné (< 1e-4 rel.)",
          (max(stredy) - min(stredy)) / abs(stredy[1]) < 1e-4, f"Δf = {[f'{v:.5f}' for v in stredy]} Hz")
    check("velká vzdálenost: Δf < 0 (přitažlivé)", df_atoms(row, xc, 2.0e-9) < 0,
          f"Δf(2 nm) = {df_atoms(row, xc, 2.0e-9):.5f} Hz")

    # --- 4. boční rozlišení vychází z dosahu síly: kontrast klesá s výškou ------
    def kontrast(d_):
        xs = np.linspace(float(row.x[7]), float(row.x[8]), 31)
        v = [df_atoms(row, x, d_) for x in xs]
        return max(v) - min(v)
    kontrasty = [kontrast(d_) for d_ in (0.7e-9, 0.85e-9, 1.0e-9, 1.3e-9)]
    check("kontrast Δf(x) přes jednu periodu klesá s rostoucí výškou (emerguje z dosahu síly)",
          all(a > b for a, b in zip(kontrasty, kontrasty[1:])),
          "p-v = " + ", ".join(f"{k:.3f}" for k in kontrasty) + " Hz při 0,70/0,85/1,00/1,30 nm")

    # --- 5. detekce atomu mimo pozici přímo nad ním -----------------------------
    izolovany = AtomRow(0.0, 1e-9, 1)
    xi = float(izolovany.x[0])
    d_set = 0.80e-9
    df_nad = df_atoms(izolovany, xi, d_set)
    df_mimo = df_atoms(izolovany, xi + SPACING / 2, d_set)
    check("izolovaný atom je vidět nad SIGMA_DF i v bočním odstupu spacing/2 od středu",
          abs(df_mimo) > SIGMA_DF,
          f"|Δf| nad atomem {abs(df_nad):.3f} Hz, o {SPACING / 2 * 1e9:.1f} nm vedle {abs(df_mimo):.3f} Hz, šum {SIGMA_DF:.3f} Hz")
    offset = np.linspace(0.0, 3e-9, 301)
    df_off = np.array([df_atoms(izolovany, xi + o, d_set) for o in offset])
    k_max = int(np.argmax(offset > 1.5e-9))
    check("|Δf| klesá s bočním odstupem od atomu monotónně (do 1,5 nm)",
          bool(np.all(np.diff(np.abs(df_off[:k_max])) <= 1e-12)),
          f"|Δf| od {abs(df_off[0]):.3f} Hz do {abs(df_off[k_max]):.4f} Hz při 1,5 nm")
    check("daleko bokem (3 nm) signál mizí (< 1e-3 hodnoty nad atomem); svislá složka síly může "
          "tam změnit znaménko, ale zanedbatelně",
          abs(df_off[-1]) < 1e-3 * abs(df_off[0]), f"|Δf(3 nm)| = {abs(df_off[-1]):.2e} Hz")

    # --- 6. zpětná kompatibilita run_loop ----------------------------------------
    def spust(force_fn, force_fn_xz, surface, row_=None, n_kroku=800, x_start=0.5e-9):
        d_set_ = 0.8e-9
        force_ref = (lambda z: row_.force_z(float(row_.x[len(row_.x) // 2]), z, U0, Ra)) if row_ else force_fn
        k_eff = kappa_eff(d_set_, A, K_CANT, F0, force_ref)
        ctrl = PIController(y0=d_set_, K_I=gain_from_tau(k_eff, 1e-3), K_P=5e-11)
        plant = FirstOrderPlant(z0=d_set_, T_sys=100e-6)
        df_set = frequency_shift(d_set_, A, K_CANT, F0, force_ref)
        return run_loop(ctrl, plant, surface, 65e-9, force_fn, A, K_CANT, F0, df_set, r_min(Ra) + A,
                        1e-5, n_kroku * 1e-5, x0=x_start, force_fn_xz=force_fn_xz)
    fn = lambda r: lj_force(r, U0, Ra)
    a = spust(fn, None, lambda x: 0.0)
    b = spust(fn, lambda x, z: fn(z), lambda x: 0.0)
    check("run_loop: force_fn_xz = F(z) dává totéž co force_fn (zpětná kompatibilita)",
          bool(np.allclose(a.z_tip, b.z_tip, rtol=0, atol=1e-18) and np.allclose(a.df, b.df, rtol=0, atol=1e-12)),
          f"{len(a.z_tip)} kroků")

    # --- 7. sken přes řadu: hrot je nad atomy výš než mezi nimi ------------------
    radek = AtomRow(1e-9, SPACING, 6)
    n_kroku = int(6 * SPACING / 65e-9 / 1e-5)
    res = spust(fn, lambda x, z: radek.force_z(x, z, U0, Ra), lambda x: 0.0, row_=radek, n_kroku=n_kroku,
                x_start=1e-9)
    x, zt = np.array(res.x), np.array(res.z_tip)
    # Sken začíná na začátku řady (1 nm): daleko od bodových atomů je síla nulová a regulátor
    # by hrot spustil až na náraz.
    nad = [np.mean(zt[np.abs(x - xc_) < 0.04e-9]) for xc_ in radek.x[1:-1]]
    mezi = [np.mean(zt[np.abs(x - (xc_ + SPACING / 2)) < 0.04e-9]) for xc_ in radek.x[1:-2]]
    check("sken bez nárazu a bez ztráty stability", not res.crashed and not res.unstable,
          f"d_min = {min(res.d) * 1e9:.3f} nm")
    check("obraz: hrot je nad atomy výš než mezi nimi (korugace > 5 pm)",
          np.mean(nad) - np.mean(mezi) > 5e-12,
          f"nad atomy {np.mean(nad) * 1e9:.4f} nm, mezi {np.mean(mezi) * 1e9:.4f} nm, korugace {(np.mean(nad) - np.mean(mezi)) * 1e12:.1f} pm")

    # --- 8. skenovací linka vedle řady --------------------------------------------
    vedle = AtomRow(0.0, SPACING, 15, y_offset=0.3e-9)
    xs = np.linspace(float(vedle.x[7]), float(vedle.x[8]), 31)
    v = [df_atoms(vedle, x_, 0.8e-9) for x_ in xs]
    p_v_vedle = max(v) - min(v)
    p_v_nad = kontrast(0.8e-9)
    check("linka posunutá o 0,3 nm vedle řady: atomy stále dávají periodický signál (p-v > 0), slabší než nad nimi",
          0.0 < p_v_vedle < p_v_nad, f"p-v vedle {p_v_vedle:.3f} Hz, nad řadou {p_v_nad:.3f} Hz")

    print()
    if all(checks):
        print(f"Souhrn: {len(checks)}/{len(checks)} [OK]")
    else:
        print(f"Souhrn: {sum(checks)}/{len(checks)} [OK], {len(checks) - sum(checks)} [SELHALO]")


if __name__ == "__main__":
    main()
