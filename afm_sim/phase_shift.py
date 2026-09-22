"""Fázový posun φ pro AM (fázovou) detekci v dynamickém AFM (kap. 2, 6, 14, 17
Voigtländer) - dvě NEZÁVISLÉ cesty výpočtu (A, B), aby šlo posoudit velikost
chyby harmonic-balance aproximace (viz README/CLAUDE.md - "obě rekonstrukce
nejsou nezávislé" je past, které se tu záměrně vyhýbáme), plus třetí cesta
(C, PLL), která naopak Δf z Cesty A záměrně SDÍLÍ - proto Cestu A nemůže
validovat, jen ukázat jinou fyzikální cestu k téže informaci (viz níže).

- resonance_phase(): Cesta A, analytická. Použije stejný harmonic-balance
  integrál jako afm_sim.frequency_shift.frequency_shift() (rovnice 17.15,
  platí pro libovolnou amplitudu A) k výpočtu Δf(d,A), a dosadí posunutou
  rezonanci do PŘESNÉHO vzorce pro fázi tlumeného buzeného oscilátoru
  (rovnice 2.20/2.28) - bez lineární aproximace "blízko rezonance" (14.16).
- phase_shift_numeric(): Cesta B, numerická ("ground truth"). Přímá integrace
  pohybové rovnice v čase s nelineární silou F(d+z), amplituda a fáze se
  extrahují dvoukanálovou lock-in demodulací (kap. 6, rovnice 6.2). Nepředpokládá
  linearizaci síly ani harmonic-balance - je to nezávislý "měřicí" kanál.

Obě cesty sdílejí jen fyzikální předpoklad "žádná disipativní interakce
hrot-vzorek" (tlumení γ = ω0/Q dané jen vlastním Q cantileveru) - stejně jako
komentář v run_simulation_afm.py o simulaci bez disipace.

- pll_phase_shift(): Cesta C, fázový závěs (PLL, obr. 17.8 - self-excitation
  mode). Nezávisí na frequency_shift() jinak než přes stejný vstupní Δf
  (harmonic-balance integrál jako u Cesty A) - simuluje vlastní dynamiku
  fázového regulačního obvodu, ne mechaniku cantileveru. Ukazuje jinou
  fyzikální cestu k téže informaci (Δf) a demonstruje diskutovaný fakt, že
  u zamčeného PLL nese informaci o vzdálenosti řídicí napětí V_Δω (tedy Δf),
  ne fáze samotná (ta je z definice skoro konstantní, chyba regulace).
"""

import numpy as np

from .frequency_shift import frequency_shift


def resonance_phase(d, A, k_cant, f0, Q, force_fn, n_theta=400):
    """Cesta A: fáze φ(d) [rad] z Δf dosazeného do přesného rezonančního vzorce.

    Buzeno na volné rezonanci ω_drive = 2π·f0 (typické pro AM mód). Vlastní
    frekvence posunutá interakcí je ω0,eff = 2π·(f0+Δf(d,A)), tlumení
    γ = ω0,eff/Q. Použije se komplexní řešení tlumeného buzeného oscilátoru
    (2.20): ẑ = ω0,eff² / (ω0,eff² - ω_drive² + i·γ·ω_drive), fáze φ = arg(ẑ)
    (2.22) - np.angle() dá rovnou správnou větev v (-π, π], na rozdíl od
    ručního rozlišování kvadrantů u (2.28).

    Args:
        d: střední (rovnovážná) poloha oscilace [m].
        A: amplituda oscilace cantileveru [m].
        k_cant: tuhost cantileveru [N/m].
        f0: volná rezonanční frekvence cantileveru [Hz].
        Q: faktor kvality cantileveru (jen vlastní tlumení, bez tip-sample
            disipace).
        force_fn: funkce F(r) -> síla hrot-vzorek [N].
        n_theta: počet bodů mřížky pro numerickou integraci Δf (viz
            frequency_shift.frequency_shift).

    Returns:
        Fáze φ [rad], typicky blízko -π/2 na monotónní (přitažlivé) větvi.
    """
    df = frequency_shift(d, A, k_cant, f0, force_fn, n_theta)
    omega0_eff = 2.0 * np.pi * (f0 + df)
    omega_drive = 2.0 * np.pi * f0
    gamma = omega0_eff / Q
    z_hat = omega0_eff**2 / (omega0_eff**2 - omega_drive**2 + 1j * gamma * omega_drive)
    return np.angle(z_hat)


def phase_shift_numeric(d, A, k_cant, f0, Q, force_fn,
                         n_periods_transient=None, n_periods_measure=20,
                         n_steps_per_period=200, return_trace=False):
    """Cesta B: (amplituda, fáze) numerickou integrací v čase + lock-in demodulací.

    Integruje RK4 rovnici pohybu (viz modulový docstring)

        z'' = -γ·z' - ω0²·(z - z_drive(t)) + F(d+z)/m,

    kde ω0 = 2π·f0, γ = ω0/Q, m = k_cant/ω0², z_drive(t) = A_drive·cos(ω0·t)
    (buzeno na volné rezonanci, stejně jako v resonance_phase). Prvních
    n_periods_transient period se zahodí (ustálení trvá ~Q period, kap. 14.5,
    obr. 14.8) - proto výchozí hodnota škáluje s Q. Na zbytku (n_periods_measure
    period) se spočítá dvoukanálová lock-in demodulace (6.2):

        X = 2·⟨z(t)·cos(ω0·t)⟩, Y = -2·⟨z(t)·sin(ω0·t)⟩

    pro z(t) = A·cos(ω0·t + φ) dá X = A·cos φ, Y = A·sin φ (ověřeno rozepsáním
    součinu přes součtové vzorce - druhý, "sumový" člen se časovým průměrem
    vynuluje, zůstane jen DC část).

    A je (na rozdíl od budicí výchylky) skutečná OSCILAČNÍ amplituda, stejný
    význam jako A v resonance_phase/frequency_shift - důležité pro srovnání
    obou cest na stejném pracovním bodě. Budicí amplituda A_drive se dopočítá
    z volné rezonanční relace A(ω0) = Q·A_drive (2.29): A_drive = A/Q. To je
    aproximace (přesná jen bez tip-sample síly), ale chyba je řádu poměru
    F'(d)/k_cant, který je v praxi ~1e-6 - 1e-3, tedy zanedbatelná oproti
    ostatním aproximacím v modelu.

    Args:
        d: střední vzdálenost hrot-vzorek [m].
        A: cílová oscilační amplituda [m] (stejný význam jako v resonance_phase).
        k_cant, f0, Q, force_fn: viz resonance_phase.
        n_periods_transient: počet period k zahození. None -> 3·Q (bezpečná
            rezerva nad "ustálení po ~Q oscilacích").
        n_periods_measure: počet period v měřicím okně pro demodulaci.
        n_steps_per_period: počet kroků RK4 integrace na jednu periodu.
            Výchozích 200 dává systematickou chybu fáze řádu 1e-3° (RK4
            diskretizace) - ověřeno na volném oscilátoru (F=0), kde je
            přesná odpověď φ=-90° známa analyticky; při 40 krocích/periodu
            je chyba už ~0.15°, srovnatelná s malými Δφ blízko d_set.
        return_trace: pokud True, navíc vrátí časovou řadu (t, z) z měřicího
            okna (po odeznění tranzientu) - pro vizualizaci lock-in demodulace.

    Returns:
        (A_meas, phi_meas), nebo (A_meas, phi_meas, t_arr, z_arr) pokud
        return_trace=True.
    """
    if n_periods_transient is None:
        n_periods_transient = max(int(3 * Q), 20)

    omega0 = 2.0 * np.pi * f0
    gamma = omega0 / Q
    m = k_cant / omega0**2
    A_drive = A / Q
    T = 1.0 / f0
    dt = T / n_steps_per_period

    def accel(z, v, t):
        z_drive = A_drive * np.cos(omega0 * t)
        return -gamma * v - omega0**2 * (z - z_drive) + force_fn(d + z) / m

    n_transient = int(n_periods_transient * n_steps_per_period)
    n_measure = int(n_periods_measure * n_steps_per_period)

    z, v, t = 0.0, 0.0, 0.0
    z_arr = np.empty(n_measure)
    t_arr = np.empty(n_measure)

    for i in range(n_transient + n_measure):
        k1z, k1v = v, accel(z, v, t)
        k2z, k2v = v + 0.5 * dt * k1v, accel(z + 0.5 * dt * k1z, v + 0.5 * dt * k1v, t + 0.5 * dt)
        k3z, k3v = v + 0.5 * dt * k2v, accel(z + 0.5 * dt * k2z, v + 0.5 * dt * k2v, t + 0.5 * dt)
        k4z, k4v = v + dt * k3v, accel(z + dt * k3z, v + dt * k3v, t + dt)
        z = z + dt / 6.0 * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
        v = v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        t = t + dt
        if i >= n_transient:
            idx = i - n_transient
            z_arr[idx] = z
            t_arr[idx] = t

    cos_ref = np.cos(omega0 * t_arr)
    sin_ref = np.sin(omega0 * t_arr)
    dt_meas = t_arr[-1] - t_arr[0]
    X = 2.0 * np.trapz(z_arr * cos_ref, t_arr) / dt_meas
    Y = -2.0 * np.trapz(z_arr * sin_ref, t_arr) / dt_meas

    A_meas = float(np.hypot(X, Y))
    phi_meas = float(np.arctan2(Y, X))
    if return_trace:
        return A_meas, phi_meas, t_arr, z_arr
    return A_meas, phi_meas


def pll_phase_shift(d, A, k_cant, f0, Q, force_fn,
                     K_pd=1.0, K_vco=2.0 * np.pi * 2e3,
                     Kp_pll=0.5, Ki_pll=800.0,
                     t_end=5e-3, dt=2.5e-6, n_theta=400, return_trace=False):
    """Cesta C: (Δf, φ, locked) z numerické simulace fázového závěsu (PLL).

    Cantilever je already self-buzený přesně na své okamžité rezonanci
    (self-excitation mode, obr. 17.8 Voigtländer) - PLL VCO se tedy jen
    snaží dohnat konstantní úhlovou frekvenci ω_cant = 2π·(f0+Δf(d,A)), kde
    Δf je TÝŽ harmonic-balance integrál jako v resonance_phase()/Cestě A
    (žádná nová závislost na mechanické ODE z Cesty B - to by vyžadovalo
    zapojit VCO zpátky do pohybové rovnice, PLL tracking mode, mimo rozsah
    tohoto minimálního rozšíření).

    Model je čistě fázově-doménový (žádné surové signály, žádné rozlišování
    jednotlivých period f0 - proto o řády levnější než phase_shift_numeric):

        e = φ_vco - φ_cant - π/2        (0 ve stavu zámku, odpovídá φ₀=90°
                                          z rovnic 17.30-17.35 v knize)
        V_phase = -K_pd·sin(e)          (fázový detektor po dolní propusti)
        V_dw = Kp_pll·V_phase + Ki_pll·∫V_phase dt   (PI regulátor smyčky)
        ω_vco = ω_work + K_vco·V_dw,   ω_work = 2π·f0            (VCO)
        de/dt = ω_vco - ω_cant

    Integruje se RK4 (stavy e a integrál PI), start z e(0)=-π/2 (obě fáze na
    nule) tak, aby test skutečně prověřil dynamiku zamykání, ne jen výchozí
    stav. Ve stavu zámku (I-složka PI eliminuje ustálenou odchylku e->0)
    platí V_dw -> Δf(d,A)·2π/K_vco přesně (matematický důsledek integrální
    akce, ne náhoda) - tedy df_pll -> Δf a phi_pll -> 0. To je očekávaný a
    poučný výsledek, ne chyba: potvrzuje, že fáze u zamčeného PLL nenese
    informaci o vzdálenosti, jen Δf (řídicí napětí V_Δω) ji nese.

    Args:
        d, A, k_cant, f0, Q, force_fn, n_theta: viz resonance_phase - Δf(d,A)
            se počítá identicky (frequency_shift), Q se dál v PLL modelu
            nepoužívá (self-excitation mode netlumí VCO vlastním Q hrotu).
        K_pd: zisk fázového detektoru [V/rad] - parametr PLL OBVODU, ne
            fyziky vzorku (ilustrativní volba, stejně jako K_P v
            run_simulation_afm.py je "NEMĚŘENÝ ilustrativní odhad").
        K_vco: zisk VCO [rad/s na V] - určuje záchytný rozsah smyčky
            (~K_vco·K_pd/2π v Hz); výchozích 2π·2 kHz stačí pokrýt typická
            Δf v tomto modelu.
        Kp_pll, Ki_pll: zisky PI regulátoru smyčky - výchozí hodnoty dávají
            přibližně kriticky tlumenou smyčku (ζ≈0.99) s vlastní frekvencí
            ~500 Hz, tedy ustálení v řádu jednotek ms.
        t_end: doba simulace [s] (výchozí 5 ms, ~5x doba ustálení výchozí
            smyčky).
        dt: krok RK4 [s] (výchozí 2,5 µs, ~800 kroků na periodu vlastní
            frekvence smyčky - bezpečná rezerva pro RK4).
        return_trace: pokud True, navíc vrátí časovou řadu (t, e) celé
            simulace (ne jen posledních 10 %) - pro vizualizaci zamykacího
            tranzientu (obdoba obr. 14.8, ale na úrovni PLL obvodu).

    Returns:
        (df_pll, phi_pll, locked): df_pll [Hz] odvozené z ustáleného V_Δω,
        phi_pll [rad] zprůměrovaná chyba fáze v posledních 10 % simulace
        (očekává se blízko 0 při zámku), locked (bool) True pokud |e| v
        posledních 10 % kroků kleslo pod 1e-2 rad (~0,57°). Pokud
        return_trace=True, navíc (t_arr, e_arr) [s, rad] s krokem dt od
        t=dt do t=t_end.
    """
    df = frequency_shift(d, A, k_cant, f0, force_fn, n_theta)
    omega_work = 2.0 * np.pi * f0
    omega_cant = 2.0 * np.pi * (f0 + df)

    def deriv(e, integ):
        v_phase = -K_pd * np.sin(e)
        v_dw = Kp_pll * v_phase + Ki_pll * integ
        omega_vco = omega_work + K_vco * v_dw
        return omega_vco - omega_cant, v_phase

    n_steps = max(int(round(t_end / dt)), 1)
    n_tail = max(n_steps // 10, 1)
    tail_start = n_steps - n_tail

    e, integ = -np.pi / 2.0, 0.0
    e_tail = np.empty(n_tail)
    if return_trace:
        t_arr = np.arange(1, n_steps + 1) * dt
        e_arr = np.empty(n_steps)

    for i in range(n_steps):
        k1e, k1i = deriv(e, integ)
        k2e, k2i = deriv(e + 0.5 * dt * k1e, integ + 0.5 * dt * k1i)
        k3e, k3i = deriv(e + 0.5 * dt * k2e, integ + 0.5 * dt * k2i)
        k4e, k4i = deriv(e + dt * k3e, integ + dt * k3i)
        e = e + dt / 6.0 * (k1e + 2.0 * k2e + 2.0 * k3e + k4e)
        integ = integ + dt / 6.0 * (k1i + 2.0 * k2i + 2.0 * k3i + k4i)
        if i >= tail_start:
            e_tail[i - tail_start] = e
        if return_trace:
            e_arr[i] = e

    v_phase_final = -K_pd * np.sin(e)
    v_dw_final = Kp_pll * v_phase_final + Ki_pll * integ
    df_pll = float(K_vco * v_dw_final / (2.0 * np.pi))
    phi_pll = float(np.mean(e_tail))
    locked = bool(np.max(np.abs(e_tail)) < 1e-2)

    if return_trace:
        return df_pll, phi_pll, locked, t_arr, e_arr
    return df_pll, phi_pll, locked
