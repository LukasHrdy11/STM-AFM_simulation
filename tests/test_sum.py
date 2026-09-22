"""Korektnostní test šumu v simulaci (stm_sim/noise.py + zapojení v run_loop).

Není to pytest test: skript vypíše [OK]/[SELHALO] pro každou kontrolu
a na konci souhrn (stejně jako test_fig_5_11.py). Čísla, proti kterým
se kontroluje, jsou z export/noise_proud/souhrn.md a export/noise/souhrn.md
(převzatá v konstantách stm_sim/noise.py).

Na konci jsou diagnostiky bez [OK]/[SELHALO]: srovnání simulace s
linearizovaným diskrétním přenosem (stm_sim/transfer.py) a mez K_P z pólů.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import run_simulation as rs
from stm_sim import noise
from stm_sim.controller import PIController, gain_from_tau
from stm_sim.plant import FirstOrderPlant
from stm_sim.sim import run_loop
from stm_sim.transfer import (
    band_rms, kp_stability_limit, psd_welch, sensitivity,
)

dt = rs.dt                     # 2 µs, stejně jako simulace
K_I = gain_from_tau(rs.kappa, rs.tau)
MEZERA_NA_PROUD = 2.0 * rs.kappa * rs.I_set   # A/m, přepočet dI -> mezera


def rel(a, b):
    """Relativní odchylka a od b."""
    return abs(a / b - 1.0)


def run_flat(t_end, current_noise=None, gap_noise=None, K_P=rs.K_P, v=rs.v, x0=0.0,
             controller=None, plant=None):
    """Běh smyčky nad rovným povrchem (H = 0) s parametry z run_simulation.py."""
    return run_loop(
        controller=controller or PIController(y0=rs.g_set, K_I=K_I, K_P=K_P),
        plant=plant or FirstOrderPlant(z0=rs.g_set, T_sys=rs.T_SYS),
        surface_fn=lambda x: 0.0, v=v, kappa=rs.kappa, V=rs.V,
        I_set=rs.I_set, g_set=rs.g_set, g_contact=rs.g_contact, dt=dt,
        t_end=t_end, x0=x0, current_noise=current_noise, gap_noise=gap_noise,
    )


def line_rms(f, psd, f0, half_width=40.0):
    """rms ostré čáry: výkon v okolí f0 minus podlaha (medián okolí)."""
    near = (f >= f0 - half_width) & (f <= f0 + half_width)
    ring = (np.abs(f - f0) > half_width) & (np.abs(f - f0) < 4 * half_width)
    floor = np.median(psd[ring])
    return math.sqrt(max(psd[near].sum() - floor * near.sum(), 0.0) * (f[1] - f[0]))


def main():
    checks = []

    def check(description, ok, detail):
        checks.append(ok)
        print(f"[{'OK' if ok else 'SELHALO'}] {description}   ({detail})")

    # --- 1. vypnutý šum = beze změny ---------------------------------------
    ramp = rs.build_surface()
    kw = dict(surface_fn=ramp, v=rs.v, kappa=rs.kappa, V=rs.V, I_set=rs.I_set,
              g_set=rs.g_set, g_contact=rs.g_contact, dt=dt, t_end=rs.t_end)
    r0 = run_loop(controller=rs.build_controller(),
                  plant=FirstOrderPlant(z0=rs.g_set, T_sys=rs.T_SYS), **kw)
    rng = np.random.default_rng(0)
    r1 = run_loop(controller=rs.build_controller(),
                  plant=FirstOrderPlant(z0=rs.g_set, T_sys=rs.T_SYS),
                  current_noise=noise.LowpassWhiteNoise(rng, 0.0, noise.F_PRE),
                  gap_noise=noise.LowpassWhiteNoise(rng, 0.0, 2000.0), **kw)
    check("Šum nulové amplitudy dá bit po bitu stejné g a e jako bez šumu",
          r0.g == r1.g and r0.e == r1.e, f"{len(r0.g)} kroků")

    # --- 2. šum proudu, bílá část ------------------------------------------
    rng = np.random.default_rng(1)
    n = 2 ** 20
    x = noise.sum_proudu(rng, cary=False).samples(n, dt)
    check("Šum proudu: sigma na krok = 2,83 pA ±3 %",
          rel(x.std(), noise.SIGMA_PROUD) < 0.03, f"{x.std() * 1e12:.3f} pA")
    N = noise.SIGMA_PROUD / math.sqrt(math.pi * noise.F_PRE / 2.0)
    f, p = psd_welch(x, dt, 2 ** 14)
    m = (f >= 100) & (f <= 2000)
    asd = math.sqrt(p[m].mean())
    check("Šum proudu: ASD 100-2000 Hz = N = 0,027 pA/√Hz ±5 %",
          rel(asd, N) < 0.05, f"{asd * 1e15:.2f} fA/√Hz vs {N * 1e15:.2f}")

    # --- 3. čáry -----------------------------------------------------------
    x_c = noise.sum_proudu(rng, cary=True).samples(n, dt)
    f, p = psd_welch(x_c, dt, 2 ** 17)
    for f0, rms0 in noise.CARY_PROUD:
        r = line_rms(f, p, f0)
        check(f"Čára {f0:.0f} Hz: rms {rms0 * 1e12:.2f} pA ±5 %",
              rel(r, rms0) < 0.05, f"{r * 1e12:.3f} pA")

    # --- 4. šum mezery -----------------------------------------------------
    x_s = noise.sum_mezery_stojici(rng).samples(n, dt)
    f, p = psd_welch(x_s, dt, 2 ** 14)
    m = (f >= 100) & (f <= 500)
    asd = math.sqrt(p[m].mean() * (1 + (300.0 / noise.F_MEZERA_STOJICI) ** 2))
    check("Mezera stojící: ASD 100-500 Hz (bez vlivu propusti) = 0,07 pm/√Hz ±5 %",
          rel(asd, noise.ASD_MEZERA_STOJICI) < 0.05, f"{asd * 1e15:.1f} fm/√Hz")

    rms_cil = {"median": 10e-12, "tichy": 2.5e-12}
    for varianta, (asd10, alfa) in noise.MEZERA_SKENY.items():
        src = noise.sum_mezery_skeny(rng, varianta)
        xk = src.samples(2 ** 22, dt)
        f, p = psd_welch(xk, dt, 2 ** 19)
        m10 = (f >= 8) & (f <= 12)
        a10 = math.sqrt(p[m10].mean() / src.psd(f[m10]).mean()) * asd10
        check(f"Mezera skeny ({varianta}): ASD(10 Hz) = {asd10 * 1e12:.2f} pm/√Hz ±10 %",
              rel(a10, asd10) < 0.10, f"{a10 * 1e12:.3f} pm/√Hz")
        edges = np.logspace(1, math.log10(400), 15)
        fc, ac = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mb = (f >= lo) & (f < hi)
            fc.append(math.sqrt(lo * hi))
            ac.append(0.5 * math.log10(p[mb].mean()))
        slope = np.polyfit(np.log10(fc), ac, 1)[0]
        check(f"Mezera skeny ({varianta}): sklon ASD f^-{alfa:.2f} ±0,1",
              abs(slope + alfa) < 0.1, f"f^{slope:.2f}")
        r = band_rms(f, p, 10, 500)
        check(f"Mezera skeny ({varianta}): rms 10-500 Hz = {rms_cil[varianta] * 1e12:.1f} pm ±15 %",
              rel(r, rms_cil[varianta]) < 0.15, f"{r * 1e12:.2f} pm")

    # --- 5. návaznost na srovnávací tabulku souhrnu (0-800 Hz, v mezeře) ---
    # export/noise_proud/souhrn.md přepočítává na mezeru s KAPPA = 1e10 1/m
    # (mimo rozsah kalibrace kappa, viz export/zavislost_proudu_na_z/souhrn.md,
    # sekce Mimo rozsah) - srovnání proto musí použít tentýž předpoklad, ne
    # aktuální rs.kappa.
    MEZERA_NA_PROUD_SOUHRN = 2.0 * 1.0e10 * rs.I_set
    f, p = psd_welch(x_c, dt, 2 ** 17)
    r = band_rms(f, p, 0, 800) / MEZERA_NA_PROUD_SOUHRN
    check("Šum proudu celkem v 0-800 Hz ≈ 0,3 pm v mezeře ±30 %",
          rel(r, 0.3e-12) < 0.30, f"{r * 1e12:.3f} pm")
    f, p = psd_welch(x_s, dt, 2 ** 17)
    r = band_rms(f, p, 0, 800)
    check("Šum mezery stojící v 0-800 Hz ≈ 2 pm (horní odhad) ±15 %",
          rel(r, 2e-12) < 0.15, f"{r * 1e12:.2f} pm")

    # --- 6. rng a backward průjezd -----------------------------------------
    def fwd_bwd(seed, cary):
        rng = np.random.default_rng(seed)
        cn, gn = noise.sum_proudu(rng, cary=cary), noise.sum_mezery_stojici(rng)
        ctrl = PIController(y0=rs.g_set, K_I=K_I, K_P=rs.K_P)
        plant = FirstOrderPlant(z0=rs.g_set, T_sys=rs.T_SYS)
        a = run_flat(0.05, cn, gn, controller=ctrl, plant=plant)
        b = run_flat(0.05, cn, gn, controller=ctrl, plant=plant, v=-rs.v, x0=a.x[-1])
        return a, b

    a1, b1 = fwd_bwd(5, cary=False)
    a2, b2 = fwd_bwd(5, cary=False)
    a3, _ = fwd_bwd(6, cary=False)
    check("Stejný SEED dá identický běh, jiný SEED jiný",
          a1.g == a2.g and b1.g == b2.g and a1.g != a3.g, "forward i backward")
    nf = np.subtract(a1.I_meas, a1.I)
    nb = np.subtract(b1.I_meas, b1.I)
    k = min(len(nf), len(nb))
    corr = abs(np.corrcoef(nf[:k], nb[:k])[0, 1])
    check("Šum proudu forward a backward je nekorelovaný (|r| < 0,05)",
          corr < 0.05, f"|r| = {corr:.4f}")
    # Reference: stejné zdroje ze stejného seedu, vzorkované nepřerušeně ve
    # stejném pořadí jako v run_loop (sdílený rng -> záleží na pořadí losování).
    rng = np.random.default_rng(5)
    cn, gn = noise.sum_proudu(rng, cary=False), noise.sum_mezery_stojici(rng)
    ref = []
    for _ in range(len(nf) + len(nb)):
        gn.step(dt)
        ref.append(cn.step(dt))
    cont = np.allclose(np.concatenate([nf, nb]), ref, rtol=0, atol=1e-20)
    check("Backward šum navazuje na forward (= jeden nepřerušený průběh)",
          cont, "stav zdroje se neresetuje")

    # --- 7. pojistka I <= 0 --------------------------------------------------
    rng = np.random.default_rng(7)
    big = noise.LowpassWhiteNoise(rng, 1e-9, noise.F_PRE)
    try:
        r = run_flat(0.01, current_noise=big)
        zaporne = sum(1 for i in r.I_meas if i <= 0)
        ok = all(math.isfinite(e) for e in r.e) and zaporne > 0
        detail = f"{zaporne} kroků s I_meas <= 0, e konečné"
    except (ValueError, OverflowError) as exc:
        ok, detail = False, repr(exc)
    check("Šum proudu 1 nA při I_set = 300 pA: běh doběhne, e je konečné", ok, detail)

    print("\nVŠECHNY KONTROLY PROŠLY" if all(checks) else "\nNĚKTERÁ KONTROLA SELHALA")

    # --- diagnostiky ---------------------------------------------------------
    print("\n=== Diagnostika (bez OK/SELHALO): simulace vs. diskrétní přenos ===")
    print(f"K_P = {rs.K_P * 1e9:.2f} nm, T_sys = {rs.T_SYS * 1e6:.0f} µs, tau = {rs.tau * 1e6:.0f} µs")
    rng = np.random.default_rng(11)
    rc = run_flat(0.5, current_noise=noise.sum_proudu(rng, cary=False))
    rg = run_flat(0.5, gap_noise=noise.sum_mezery_stojici(rng))
    nseg = 2 ** 14
    f, p_in_c = psd_welch(np.subtract(rc.I_meas, rc.I) / MEZERA_NA_PROUD, dt, nseg)
    _, p_out_c = psd_welch(np.subtract(rc.g, rs.g_set), dt, nseg)
    _, p_in_g = psd_welch(np.subtract(rg.g, rg.z_tip), dt, nseg)
    _, p_out_g = psd_welch(np.subtract(rg.g, rs.g_set), dt, nseg)
    f, p_in_c, p_out_c, p_in_g, p_out_g = (a[1:] for a in (f, p_in_c, p_out_c, p_in_g, p_out_g))
    S, T = sensitivity(f, dt, rs.kappa, rs.K_P, K_I, rs.T_SYS)
    print("   f [Hz]    |T| sim  |T| teorie    |S| sim  |S| teorie")
    for f0 in [100, 300, 1000, 3000, 10000, 20000]:
        mb = (f >= f0 / 1.2) & (f <= f0 * 1.2)
        tc = math.sqrt(p_out_c[mb].mean() / p_in_c[mb].mean())
        sg = math.sqrt(p_out_g[mb].mean() / p_in_g[mb].mean())
        tt = math.sqrt((np.abs(T[mb]) ** 2).mean())
        st = math.sqrt((np.abs(S[mb]) ** 2).mean())
        print(f"   {f0:6d}   {tc:8.3f}  {tt:8.3f}    {sg:8.3f}  {st:8.3f}")
    lim = kp_stability_limit(dt, rs.kappa, K_I, rs.T_SYS)
    print(f"Mez K_P z pólů diskrétní smyčky: {lim * 1e9:.3f} nm "
          f"(empiricky změřeno 4,96 nm, viz run_simulation.py)")
    print("Nestabilní pól leží na z = -1 (Nyquist simulace) -> artefakt kroku dt.")


if __name__ == "__main__":
    main()
