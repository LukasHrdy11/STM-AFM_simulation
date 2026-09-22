"""Sestavení a spuštění FM-AFM simulace z jednoho slovníku parametrů.

Obdoba vypocet_stm.py pro AFM (viz tam pro zdůvodnění rozdělení
výpočet/widgety a pro pravidlo přenositelnosti).

Dva rozdíly proti STM, na které je potřeba myslet v UI:
- Pracovní bod d_set MUSÍ ležet za minimem Δf(d). Když ne, je citlivost
  kappa_eff záporná a smyčka nemá co regulovat; sestav_regulator() to
  hlásí výjimkou, ne tichým nesmyslem.
- GAIN_MODE "local" přepočítává citlivost v každém kroku a umí ohlásit
  ztrátu stability. To je funkce, ne chyba - přesně to má demonstrovat.
"""

import numpy as np

from afm_sim import measured, noise
from afm_sim.frequency_shift import frequency_shift, gain_from_tau, kappa_eff
from afm_sim.sim import run_loop
from afm_sim.tip_force import lj_force, r_min
from afm_sim.tip_force_atoms import AtomRow
from stm_sim.controller import (
    IntegralController, PIController, ProportionalController,
)
from stm_sim.plant import FirstOrderPlant
from stm_sim.surface import atoms_surface, ramp_surface, step_surface

VYCHOZI = dict(
    measured.PRESET_QPLUS,
    controller="PI",
    surface="atoms",
    gain_mode="fixed",      # "fixed" | "local" (local umí hlásit nestabilitu)
    backward=True,
    mera_sumu=1.0,          # násobek změřené velikosti šumu Δf i amplitudy
    seed=42,
    # geometrie povrchu
    x_edge=3e-9,
    H=0.08e-9,
    w=0.3e-9,
    atoms_start=3e-9,
    atoms_H=0.15e-9,
    atoms_spacing=0.6e-9,
    atoms_sigma=0.10e-9,
    atoms_pocet=6,
    # řada bodových atomů (surface = "atoms-AFM")
    atoms_afm_start=3e-9,
    atoms_afm_spacing=0.6e-9,
    atoms_afm_pocet=6,
    atoms_afm_y=0.0,
)


def sestav_sily(p):
    """Silový model: (force_fn, force_fn_xz, force_ref, atoms_row).

    Pro běžné povrchy je síla funkcí jen vzdálenosti. Pro "atoms-AFM"
    (řada BODOVÝCH atomů) závisí i na x - hrot atom cítí i mimo pozici
    přímo nad ním, protože se sčítají svislé složky sil ve 3D. Referenční
    síla je nad prostředním atomem (nejhorší případ, má oba sousedy).
    """
    force_fn = lambda r: lj_force(r, p["U0"], p["Ra"])
    if p["surface"] != "atoms-AFM":
        return force_fn, None, force_fn, None

    atoms_row = AtomRow(p["atoms_afm_start"], p["atoms_afm_spacing"],
                        p["atoms_afm_pocet"], y_offset=p["atoms_afm_y"])
    x_ref = float(atoms_row.x[p["atoms_afm_pocet"] // 2])
    force_fn_xz = lambda x, z: atoms_row.force_z(x, z, p["U0"], p["Ra"])
    force_ref = lambda z: atoms_row.force_z(x_ref, z, p["U0"], p["Ra"])
    return force_fn, force_fn_xz, force_ref, atoms_row


def d_minima_frequency_shift(p, force_ref, d_contact):
    """Poloha minima Δf(d) - pracovní bod musí ležet ZA ním (dál od vzorku)."""
    grid = np.linspace(d_contact, 5.0 * p["Ra"], 400)
    vals = [frequency_shift(d, p["A"], p["k_cant"], p["f0"], force_ref)
            for d in grid]
    return grid[int(np.argmin(vals))]


def sestav_regulator(p, force_ref, d_set, d_contact):
    """Regulátor pro AFM; ověří, že pracovní bod leží v použitelné větvi Δf(d)."""
    d_argmin = d_minima_frequency_shift(p, force_ref, d_contact)
    if not (d_set > d_argmin):
        raise ValueError(
            f"Pracovní bod d_set = {d_set * 1e9:.3f} nm musí ležet ZA minimem "
            f"Δf(d) (to je na {d_argmin * 1e9:.3f} nm). Blíž ke vzorku je "
            "citlivost kappa_eff záporná a smyčka by regulovala opačně - "
            "posuň d_set dál od vzorku.")
    k_eff = kappa_eff(d_set, p["A"], p["k_cant"], p["f0"], force_ref)
    K_I = gain_from_tau(k_eff, p["tau"])
    if p["controller"] == "P":
        return ProportionalController(base=d_set, K_P=p["K_P"])
    if p["controller"] == "I":
        return IntegralController(y0=d_set, K_I=K_I)
    if p["controller"] == "PI":
        return PIController(y0=d_set, K_I=K_I, K_P=p["K_P"])
    raise ValueError(f"Neznámý controller: {p['controller']!r} (očekávám P/I/PI)")


def sestav_povrch(p):
    """Funkce h(x); pro "atoms-AFM" je to rovina (výšku nese silový model)."""
    if p["surface"] == "step":
        return lambda x: step_surface(x, p["x_edge"], p["H"])
    if p["surface"] == "ramp":
        return lambda x: ramp_surface(x, p["x_edge"], p["H"], p["w"])
    if p["surface"] == "atoms":
        return lambda x: atoms_surface(x, p["atoms_start"], p["atoms_H"],
                                       p["atoms_spacing"], p["atoms_sigma"])
    if p["surface"] == "atoms-AFM":
        return lambda x: 0.0
    raise ValueError(f"Neznámý surface: {p['surface']!r} "
                     "(očekávám step/ramp/atoms/atoms-AFM)")


def sestav_sum(p):
    """Zdroje šumu (Δf, amplituda) škálované posuvníkem p['mera_sumu'].

    Šum amplitudy se losuje ze SAMOSTATNÉHO generátoru, aby jeho zapnutí
    nezměnilo šum Δf (stejně jako v run_simulation_afm.py). Šum mezery se
    nezapojuje - pro AFM nebyl změřen.
    """
    mira = p["mera_sumu"]
    freq_noise = None
    if p["sum_frekvence"] and mira > 0:
        freq_noise = noise.sum_frekvence(np.random.default_rng(p["seed"]),
                                         sigma=noise.SIGMA_DF * mira)
    amp_noise = None
    if p["sum_amplitudy"] and mira > 0:
        amp_noise = noise.sum_amplitudy(np.random.default_rng([p["seed"], 1]),
                                        sigma=noise.SIGMA_AMP * mira)
    return freq_noise, amp_noise


def delka_skenu(p):
    """Doba skenu [s] a počáteční x (řada bodových atomů začíná až u atomů)."""
    if p["surface"] == "atoms":
        konec = p["atoms_start"] + (p["atoms_pocet"] + 0.5) * p["atoms_spacing"]
        return 20 * p["tau"] + konec / p["v"], 0.0
    if p["surface"] == "atoms-AFM":
        # Sken projede JEN řadu atomů: daleko od bodových atomů je síla
        # nulová a regulátor by hrot spustil až na náraz (není tu podložka).
        return (p["atoms_afm_pocet"] * p["atoms_afm_spacing"] / p["v"],
                p["atoms_afm_start"])
    return 20 * p["tau"] + p["x_edge"] / p["v"], 0.0


def spust(parametry=None, **zmeny):
    """Spustí AFM simulaci a vrátí výsledky i kontext ke kreslení.

    Returns:
        Slovník: parametry, surface_fn, atoms_row, fwd, bwd (nebo None),
        df_set, d_contact, t_end, dt.

    Raises:
        ValueError: když pracovní bod d_set leží před minimem Δf(d)
            (sestav_regulator to hlásí srozumitelnou hláškou pro UI).
    """
    p = dict(VYCHOZI if parametry is None else parametry)
    p.update(zmeny)

    force_fn, force_fn_xz, force_ref, atoms_row = sestav_sily(p)
    d_contact = r_min(p["Ra"]) + p["A"]
    d_set = (p["d_set"] if p["surface"] != "atoms-AFM"
             else measured.D_SET_ATOMS_AFM)

    t_end, x0 = delka_skenu(p)
    dt = p["tau"] / 100
    surface_fn = sestav_povrch(p)
    df_set = frequency_shift(d_set, p["A"], p["k_cant"], p["f0"], force_ref)
    controller = sestav_regulator(p, force_ref, d_set, d_contact)
    plant = FirstOrderPlant(z0=d_set, T_sys=p["T_SYS"])
    freq_noise, amp_noise = sestav_sum(p)

    spolecne = dict(
        surface_fn=surface_fn, force_fn=force_fn, A=p["A"], k_cant=p["k_cant"],
        f0=p["f0"], df_set=df_set, d_contact=d_contact, dt=dt, t_end=t_end,
        adaptive_tau=p["tau"] if p["gain_mode"] == "local" else None,
        freq_noise=freq_noise, gap_noise=None, force_fn_xz=force_fn_xz,
        amp_noise=amp_noise,
    )

    fwd = run_loop(controller=controller, plant=plant, v=p["v"], x0=x0,
                   **spolecne)

    bwd = None
    if p["backward"] and not fwd.crashed and not fwd.unstable:
        bwd = run_loop(controller=controller, plant=plant, v=-p["v"],
                       x0=fwd.x[-1], t0=fwd.t[-1] + dt, **spolecne)

    return {"parametry": p, "surface_fn": surface_fn, "atoms_row": atoms_row,
            "fwd": fwd, "bwd": bwd, "df_set": df_set, "d_contact": d_contact,
            "d_set": d_set, "t_end": t_end, "dt": dt}


def rezonancni_krivka(p, df=0.0, n=601, sirka_q=8.0):
    """Amplitudová a fázová odezva cantileveru kolem rezonance.

    Ukazuje, co Δf vlastně znamená: posun CELÉ rezonanční křivky. Δf není
    měřená veličina, ale odečtená z toho, kam se rezonance posunula.

    Args:
        p: slovník parametrů (potřebuje k_cant, f0, A a Q).
        df: posun rezonance [Hz] (0 = volný cantilever daleko od vzorku).
        n: počet bodů křivky.
        sirka_q: šířka okna ve vzorcích f0/Q kolem rezonance.

    Returns:
        (f, amplituda, faze) - frekvence [Hz], bezrozměrná amplituda
        normovaná na statickou výchylku, fáze [stupně].
    """
    Q = p.get("Q", 5000.0)
    f0 = p["f0"]
    f_res = f0 + df
    sirka = sirka_q * f0 / Q
    f = np.linspace(f_res - sirka, f_res + sirka, n)

    # Vynucený tlumený oscilátor: A(f) = 1 / sqrt((1-r^2)^2 + (r/Q)^2),
    # faze = -atan2(r/Q, 1-r^2), kde r = f/f_res.
    r = f / f_res
    jmenovatel = np.sqrt((1 - r**2) ** 2 + (r / Q) ** 2)
    amplituda = 1.0 / jmenovatel
    faze = np.degrees(-np.arctan2(r / Q, 1 - r**2))
    return f, amplituda, faze
