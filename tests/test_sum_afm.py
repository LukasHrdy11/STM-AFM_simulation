"""Korektnostní test šumu ve FM-AFM simulaci (afm_sim/noise.py + zapojení
v run_loop).

Není to pytest test: skript vypíše [OK]/[SELHALO] pro každou kontrolu
a na konci souhrn (stejně jako test_sum.py u STM). Čísla, proti kterým se
kontroluje, jsou z export/sum_frekvence_afm/souhrn.md (převzatá v
konstantách afm_sim/noise.py).
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import run_simulation_afm as rs
from afm_sim import noise
from afm_sim.sim import run_loop
from stm_sim.controller import PIController
from stm_sim.plant import FirstOrderPlant
from stm_sim.transfer import psd_welch

dt = rs.dt


def rel(a, b):
    """Relativní odchylka a od b."""
    return abs(a / b - 1.0)


def run_flat(t_end, freq_noise=None, gap_noise=None, v=rs.v, x0=0.0,
             controller=None, plant=None):
    """Běh smyčky nad rovným povrchem (h = 0) s parametry z run_simulation_afm.py."""
    return run_loop(
        controller=controller or PIController(y0=rs.d_set, K_I=rs.gain_from_tau(
            rs.kappa_eff(rs.d_set, rs.A, rs.k_cant, rs.f0, rs.force_fn), rs.tau),
            K_P=rs.K_P),
        plant=plant or FirstOrderPlant(z0=rs.d_set, T_sys=rs.T_SYS),
        surface_fn=lambda x: 0.0, v=v, force_fn=rs.force_fn, A=rs.A,
        k_cant=rs.k_cant, f0=rs.f0, df_set=rs.frequency_shift(
            rs.d_set, rs.A, rs.k_cant, rs.f0, rs.force_fn),
        d_contact=rs.d_contact, dt=dt, t_end=t_end, x0=x0,
        freq_noise=freq_noise, gap_noise=gap_noise,
    )


def main():
    checks = []

    def check(description, ok, detail):
        checks.append(ok)
        print(f"[{'OK' if ok else 'SELHALO'}] {description}   ({detail})")

    # --- 1. vypnutý šum = beze změny ---------------------------------------
    surface_fn = rs.build_surface()
    kw = dict(surface_fn=surface_fn, v=rs.v, force_fn=rs.force_fn, A=rs.A,
              k_cant=rs.k_cant, f0=rs.f0,
              df_set=rs.frequency_shift(rs.d_set, rs.A, rs.k_cant, rs.f0, rs.force_fn),
              d_contact=rs.d_contact, dt=dt, t_end=rs.t_end)
    r0 = run_loop(controller=rs.build_controller(),
                  plant=FirstOrderPlant(z0=rs.d_set, T_sys=rs.T_SYS), **kw)
    rng = np.random.default_rng(0)
    r1 = run_loop(controller=rs.build_controller(),
                  plant=FirstOrderPlant(z0=rs.d_set, T_sys=rs.T_SYS),
                  freq_noise=noise.LowpassWhiteNoise(rng, 0.0, noise.F_C_DF), **kw)
    check("Šum nulové amplitudy dá bit po bitu stejné d a e jako bez šumu",
          r0.d == r1.d and r0.e == r1.e, f"{len(r0.d)} kroků")

    # --- 2. šum Δf, bílá část -----------------------------------------------
    rng = np.random.default_rng(1)
    n = 2 ** 20
    x = noise.sum_frekvence(rng).samples(n, dt)
    check("Šum Δf: sigma na krok = %.4f Hz ±5 %%" % noise.SIGMA_DF,
          rel(x.std(), noise.SIGMA_DF) < 0.05, f"{x.std():.4f} Hz")
    N = noise.SIGMA_DF / math.sqrt(math.pi * noise.F_C_DF / 2.0)
    f, p = psd_welch(x, dt, 2 ** 17)
    m = (f >= 0.5) & (f <= 5.0)   # hluboko pod F_C_DF = 25 Hz, ne u Nyquistu
    asd = math.sqrt(p[m].mean())
    check("Šum Δf: ASD hluboko pod F_C_DF konzistentní s N ±10 %%",
          rel(asd, N) < 0.10, f"{asd:.4f} Hz/√Hz vs {N:.4f}")

    # --- 3. rng a backward průjezd -------------------------------------------
    def fwd_bwd(seed):
        rng = np.random.default_rng(seed)
        fn = noise.sum_frekvence(rng)
        ctrl = PIController(y0=rs.d_set, K_I=rs.gain_from_tau(
            rs.kappa_eff(rs.d_set, rs.A, rs.k_cant, rs.f0, rs.force_fn), rs.tau), K_P=rs.K_P)
        plant = FirstOrderPlant(z0=rs.d_set, T_sys=rs.T_SYS)
        a = run_flat(0.05, freq_noise=fn, controller=ctrl, plant=plant)
        b = run_flat(0.05, freq_noise=fn, controller=ctrl, plant=plant, v=-rs.v, x0=a.x[-1])
        return a, b

    a1, b1 = fwd_bwd(5)
    a2, b2 = fwd_bwd(5)
    a3, _ = fwd_bwd(6)
    check("Stejný SEED dá identický běh, jiný SEED jiný",
          a1.d == a2.d and b1.d == b2.d and a1.d != a3.d, "forward i backward")

    rng = np.random.default_rng(5)
    fn = noise.sum_frekvence(rng)
    ref = [fn.step(dt) for _ in range(len(a1.d) + len(b1.d))]
    nf = np.subtract(a1.df_meas, a1.df)
    nb = np.subtract(b1.df_meas, b1.df)
    cont = np.allclose(np.concatenate([nf, nb]), ref, rtol=0, atol=1e-12)
    check("Backward šum navazuje na forward (= jeden nepřerušený průběh)",
          cont, "stav zdroje se neresetuje")

    # --- 4. pojistka - extrémní šum nezpůsobí NaN/inf ------------------------
    rng = np.random.default_rng(7)
    big = noise.LowpassWhiteNoise(rng, 1.0, noise.F_C_DF)
    try:
        r = run_flat(0.01, freq_noise=big)
        ok = all(math.isfinite(e) for e in r.e)
        detail = f"{len(r.e)} kroků, e konečné"
    except (ValueError, OverflowError) as exc:
        ok, detail = False, repr(exc)
    check("Šum Δf 1 Hz při df_set malém: běh doběhne, e je konečné", ok, detail)

    # --- 5. šum amplitudy ------------------------------------------------------
    rng = np.random.default_rng(11)
    xa = noise.sum_amplitudy(rng).samples(2 ** 20, dt)
    check("Šum amplitudy: sigma na krok = %.2f pm ±5 %%" % (noise.SIGMA_AMP * 1e12),
          rel(xa.std(), noise.SIGMA_AMP) < 0.05, f"{xa.std() * 1e12:.3f} pm")

    kw2 = dict(surface_fn=surface_fn, v=rs.v, force_fn=rs.force_fn, A=rs.A, k_cant=rs.k_cant, f0=rs.f0,
               df_set=rs.frequency_shift(rs.d_set, rs.A, rs.k_cant, rs.f0, rs.force_fn),
               d_contact=rs.d_contact, dt=dt, t_end=0.05)

    def beh(freq_seed=None, amp_seed=None, amp_sigma=noise.SIGMA_AMP):
        fn = noise.sum_frekvence(np.random.default_rng(freq_seed)) if freq_seed is not None else None
        an = (noise.sum_amplitudy(np.random.default_rng([amp_seed, 1]), sigma=amp_sigma)
              if amp_seed is not None else None)
        return run_loop(controller=rs.build_controller(),
                        plant=FirstOrderPlant(z0=rs.d_set, T_sys=rs.T_SYS),
                        freq_noise=fn, amp_noise=an, **kw2)

    r_bez, r_nula = beh(), beh(amp_seed=3, amp_sigma=0.0)
    check("Šum amplitudy nulové sigma dá bit po bitu stejný běh jako bez něj",
          r_bez.d == r_nula.d and r_bez.df == r_nula.df, f"{len(r_bez.d)} kroků")

    r_amp = beh(amp_seed=3)
    r_amp2 = beh(amp_seed=3)
    r_amp3 = beh(amp_seed=4)
    check("Šum amplitudy: stejný seed dá identický běh, jiný jiný; šum se do Δf opravdu dostane",
          r_amp.df == r_amp2.df and r_amp.df != r_amp3.df and r_amp.df != r_bez.df, "reprodukovatelné")

    a_f = beh(freq_seed=5)
    b_f = beh(freq_seed=5, amp_seed=3)
    n_a = np.subtract(a_f.df_meas, a_f.df)
    n_b = np.subtract(b_f.df_meas, b_f.df)
    check("Zapnutí šumu amplitudy nezmění realizaci šumu Δf (samostatný generátor)",
          bool(np.allclose(n_a, n_b, rtol=0, atol=1e-12)), f"{len(n_a)} vzorků šumu Δf shodných")

    # velikost vlivu: δΔf = Δf(A + δA) - Δf(A) při d_set, proti šumu Δf
    dA = noise.sum_amplitudy(np.random.default_rng(13)).samples(4000, dt)
    df0 = rs.frequency_shift(rs.d_set, rs.A, rs.k_cant, rs.f0, rs.force_fn)
    ddf = np.array([rs.frequency_shift(rs.d_set, rs.A + a, rs.k_cant, rs.f0, rs.force_fn) for a in dA]) - df0
    d_blizko = rs.r_min(rs.Ra) + rs.A + 0.05e-9    # těsně nad d_contact (nejhorší případ)
    df0b = rs.frequency_shift(d_blizko, rs.A, rs.k_cant, rs.f0, rs.force_fn)
    ddfb = np.array([rs.frequency_shift(d_blizko, rs.A + a, rs.k_cant, rs.f0, rs.force_fn) for a in dA[:800]]) - df0b
    check("Vliv šumu amplitudy na Δf je pod šumem Δf i těsně nad d_contact (informativně)",
          ddf.std() < noise.SIGMA_DF and ddfb.std() < noise.SIGMA_DF,
          f"při d_set {ddf.std():.4f} Hz, těsně nad d_contact {ddfb.std():.4f} Hz, šum Δf {noise.SIGMA_DF:.3f} Hz")

    print("\nVŠECHNY KONTROLY PROŠLY" if all(checks) else "\nNĚKTERÁ KONTROLA SELHALA")


if __name__ == "__main__":
    main()
