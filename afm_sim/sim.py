"""Hlavní simulační smyčka FM-AFM: časová integrace, detekce nárazu,
detekce nestability zesílení, záznam průběhů.

Analog stm_sim/sim.py::run_loop - stejná struktura (regulátor a akční
člen oddělené, `surface_fn` volitelný podle volajícího), rozdíl je ve
fyzice měřené veličiny (Δf místo I) a ve volitelném režimu `local`
(zesílení regulátoru se přepočítává v každém kroku podle aktuální
citlivosti kappa_eff(d), místo aby bylo pevné jako u `fixed`).
"""

import math
from dataclasses import dataclass, field

from .frequency_shift import frequency_shift, gain_from_tau, kappa_eff


@dataclass
class SimulationResult:
    t: list = field(default_factory=list)
    x: list = field(default_factory=list)
    z_tip: list = field(default_factory=list)
    d: list = field(default_factory=list)
    df: list = field(default_factory=list)
    df_meas: list = field(default_factory=list)
    e: list = field(default_factory=list)
    crashed: bool = False
    crash_index: int = None
    unstable: bool = False
    unstable_index: int = None


def run_loop(controller, plant, surface_fn, v, force_fn, A, k_cant, f0,
             df_set, d_contact, dt, t_end, x0=0.0, t0=0.0,
             freq_noise=None, gap_noise=None, adaptive_tau=None, force_fn_xz=None,
             amp_noise=None):
    """Simuluje zpětnovazební smyčku FM-AFM podél přejezdu hrotu v ose x.

    Hrot jede konstantní rychlostí v podél x. V každém kroku: spočítá se
    vzdálenost hrot-vzorek d (mean poloha oscilace, ne okamžitá výchylka -
    ta jde do d + A*sinθ uvnitř frequency_shift), z ní posun frekvence
    Δf(d) a chyba e = df_set - Δf(d) (kladné e = "moc blízko, ustupovat",
    stejná konvence jako u STM). Regulátor z e spočítá příkaz y, akční
    člen (plant) podle y aktualizuje z_tip.

    Simulace se zastaví buď při nárazu (d <= d_contact - volající musí
    d_contact zvolit tak, aby zahrnoval amplitudu oscilace, viz
    run_simulation_afm.py), nebo (v režimu `local`) při ztrátě citlivosti
    (kappa_eff(d) <= 0 za minimem Δf(d)) - odtud dál by K_I vyšlo záporné
    a smyčka by se rozeběhla, což je přesně jev, který má `local` režim
    demonstrovat, ne skrýt.

    Args:
        controller: objekt s metodou step(e, dt) -> y.
        plant: objekt s metodou step(y, dt) -> z_tip a atributem z.
        surface_fn: funkce h(x) -> výška povrchu [m].
        v: rychlost pojezdu hrotu [m/s].
        force_fn: funkce F(r) -> síla hrot-vzorek [N] (viz tip_force.lj_force).
        A: amplituda oscilace cantileveru [m].
        k_cant: tuhost cantileveru [N/m].
        f0: rezonanční frekvence cantileveru [Hz].
        df_set: požadovaný posun frekvence (setpoint) [Hz].
        d_contact: vzdálenost (mean poloha), při které se hlásí náraz [m].
        dt: délka časového kroku [s].
        t_end: konec simulace [s].
        x0: počáteční poloha x tohoto průjezdu [m].
        t0: počáteční čas tohoto průjezdu pro záznam do result.t [s].
        freq_noise: zdroj šumu měřeného Δf se step(dt), nebo None (šum
            zatím není implementován, parametr je jen pro budoucí použití
            se stejným rozhraním jako u STM).
        gap_noise: zdroj šumu vzdálenosti d se step(dt), nebo None (viz
            freq_noise).
        force_fn_xz: None (výchozí) -> síla závisí jen na svislé vzdálenosti
            d = z_tip - h(x), použije se `force_fn`. Funkce F_z(x, z) -> síla [N]
            -> síla závisí i na poloze x hrotu (povrch z bodových atomů, viz
            afm_sim/tip_force_atoms.py); `force_fn` se pak ignoruje, `z` je výška
            hrotu nad rovinou atomů, takže `surface_fn` má být h(x) = 0
            (d = z_tip). Zpětně kompatibilní: bez tohoto parametru se chování nemění.
        amp_noise: zdroj šumu amplitudy oscilace se step(dt) [m], nebo None (bez šumu).
            V každém kroku se k A přičte jeho hodnota a použije se v Δf i v kappa_eff.
            Doporučeno losovat z jiného generátoru než šum Δf, aby jeho zapnutí
            nezměnilo realizaci šumu Δf (viz afm_sim/noise.py).
        adaptive_tau: None -> GAIN_MODE "fixed" (K_I regulátoru se
            nemění). Float tau -> GAIN_MODE "local": před každým
            controller.step(e, dt) se přepočte
            controller.K_I = gain_from_tau(kappa_eff(d, ...), tau).

    Returns:
        SimulationResult se zaznamenanými průběhy a příznaky nárazu/nestability.
    """
    result = SimulationResult()

    n_steps = int(t_end / dt)
    z_tip = plant.z
    for i in range(n_steps):
        t_local = i * dt
        t = t0 + t_local
        x = x0 + v * t_local
        h = surface_fn(x)
        d = z_tip - h
        if gap_noise is not None:
            d += gap_noise.step(dt)
        A_krok = A if amp_noise is None else A + amp_noise.step(dt)
        if force_fn_xz is None:
            force_krok = force_fn
        else:
            force_krok = lambda z, x=x: force_fn_xz(x, z)
        df = frequency_shift(d, A_krok, k_cant, f0, force_krok)
        if freq_noise is None:
            df_meas = df
        else:
            df_meas = df + freq_noise.step(dt)
        e = df_set - df_meas

        result.t.append(t)
        result.x.append(x)
        result.z_tip.append(z_tip)
        result.d.append(d)
        result.df.append(df)
        result.df_meas.append(df_meas)
        result.e.append(e)

        if d <= d_contact:
            result.crashed = True
            result.crash_index = i
            break

        if adaptive_tau is not None:
            k_eff = kappa_eff(d, A_krok, k_cant, f0, force_krok)
            if k_eff <= 0:
                result.unstable = True
                result.unstable_index = i
                break
            controller.K_I = gain_from_tau(k_eff, adaptive_tau)

        y = controller.step(e, dt)
        z_tip = plant.step(y, dt)

    return result


@dataclass
class ConstantHeightResult:
    """Výsledek skenu s VYPNUTOU zpětnou vazbou (režim konstantní výšky)."""
    t: list = field(default_factory=list)
    x: list = field(default_factory=list)
    z_tip: list = field(default_factory=list)
    d: list = field(default_factory=list)
    df: list = field(default_factory=list)
    df_meas: list = field(default_factory=list)
    crashed: bool = False
    crash_index: int = None


def constant_height_scan(surface_fn, v, force_fn, A, k_cant, f0, d_contact,
                         dt, t_end, z_fixed, x0=0.0, t0=0.0, freq_noise=None,
                         gap_noise=None, force_fn_xz=None, amp_noise=None):
    """Sken v konstantní výšce: hrot jede v pevném z, regulátor je vypnutý.

    Obdoba stm_sim.sim.constant_height_scan pro FM-AFM. Nic se neintegruje -
    hrot jede rychlostí v podél x ve stále stejné výšce z_fixed a v každém
    kroku se jen spočítá, jaké Δf by na té pozici bylo. Topografie se tím
    promítne přímo do Δf, místo aby ji smyčka vykompenzovala pohybem hrotu.

    Δf se počítá týmž vzorcem (rovnice 17.15) jako v run_loop(), takže oba
    režimy popisují týž hrot a jdou vykreslit do jednoho grafu.

    Pozor na rozdíl proti STM: Δf(d) je NEMONOTÓNNÍ (viz frequency_shift.py).
    Při přiblížení k povrchu proto Δf nejdřív klesá, za minimem zase roste -
    bez regulátoru jde hrot tím minimem projet, což se se zapnutou smyčkou
    projeví jako ztráta stability. Tady se jen zaznamená průběh; kritérium
    nárazu je stejné (d <= d_contact).

    Args:
        surface_fn: funkce h(x) -> výška povrchu [m].
        v: rychlost pojezdu hrotu [m/s] (záporná = proti ose x).
        force_fn: funkce F(r) -> síla hrot-vzorek [N].
        A: amplituda oscilace cantileveru [m].
        k_cant: tuhost cantileveru [N/m].
        f0: rezonanční frekvence cantileveru [Hz].
        d_contact: vzdálenost, při které se hlásí náraz [m].
        dt: délka časového kroku [s].
        t_end: konec skenu [s].
        z_fixed: pevná výška hrotu [m] - jediný "ovladač" tohoto režimu.
        x0: počáteční poloha x [m].
        t0: počáteční čas pro záznam [s].
        freq_noise: zdroj šumu měřeného Δf [Hz] se step(dt), nebo None.
        gap_noise: zdroj šumu vzdálenosti [m] se step(dt), nebo None.
        force_fn_xz: síla závislá i na x (řada bodových atomů), nebo None.
        amp_noise: zdroj šumu amplitudy [m] se step(dt), nebo None.

    Returns:
        ConstantHeightResult. Chyba regulátoru se nezaznamenává - není co
        regulovat.
    """
    result = ConstantHeightResult()

    n_steps = int(t_end / dt)
    for i in range(n_steps):
        t_local = i * dt
        t = t0 + t_local
        x = x0 + v * t_local
        h = surface_fn(x)
        d = z_fixed - h
        if gap_noise is not None:
            d += gap_noise.step(dt)
        A_krok = A if amp_noise is None else A + amp_noise.step(dt)
        if force_fn_xz is None:
            force_krok = force_fn
        else:
            force_krok = lambda z, x=x: force_fn_xz(x, z)
        df = frequency_shift(d, A_krok, k_cant, f0, force_krok)
        df_meas = df if freq_noise is None else df + freq_noise.step(dt)

        result.t.append(t)
        result.x.append(x)
        result.z_tip.append(z_fixed)
        result.d.append(d)
        result.df.append(df)
        result.df_meas.append(df_meas)

        if d <= d_contact:
            result.crashed = True
            result.crash_index = i
            break

    return result
