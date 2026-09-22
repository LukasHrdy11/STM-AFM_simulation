"""Hlavní simulační smyčka: časová integrace, detekce nárazu, záznam průběhů.

Jediná smyčka run_loop() je společná pro libovolný regulátor (P/I/PI,
viz controller.py), libovolný akční člen (FirstOrderPlant, viz plant.py)
a libovolný povrch (surface_fn, viz surface.py). Konkrétní volbu dělá
volající (typicky run_simulation.py).

Šum (volitelný, výchozí vypnuto) dodávají zdroje z noise.py: šum proudu se
přičítá k MĚŘENÉMU proudu, šum mezery k fyzické mezeře g.
"""

import math
from dataclasses import dataclass, field

from .tip_current import c_from_setpoint, current

# Dolní mez |I_meas| pro log-chybu (jen proti log(0) při extrémním šumu).
I_MIN = 1e-15  # A


@dataclass
class SimulationResult:
    t: list = field(default_factory=list)
    x: list = field(default_factory=list)
    z_tip: list = field(default_factory=list)
    g: list = field(default_factory=list)
    I: list = field(default_factory=list)
    I_meas: list = field(default_factory=list)
    e: list = field(default_factory=list)
    crashed: bool = False
    crash_index: int = None


def run_loop(controller, plant, surface_fn, v, kappa, V, I_set, g_set,
             g_contact, dt, t_end, x0=0.0, t0=0.0, current_noise=None,
             gap_noise=None, callback=None):
    """Simuluje zpětnovazební smyčku STM podél přejezdu hrotu v ose x.

    Hrot jede konstantní rychlostí v podél x, počínaje pozicí x0. V každém
    kroku: změří se proud, z něj se spočítá log-chyba
    e, regulátor z e spočítá příkaz y a akční člen (plant) podle y
    aktualizuje skutečnou polohu hrotu z_tip. Simulace se zastaví v
    okamžiku prvního nárazu (g <= g_contact), protože model za nárazem
    neplatí.

    Předáním záporného v a x0 rovného koncové poloze z předchozího volání
    (spolu se stejnou instancí controller/plant) lze navázat zpětný
    (backward) průjezd stejnou linkou - regulátor i poloha hrotu pak
    pokračují ze stavu, kde skončil předchozí (forward) průjezd, přesně
    jako u reálného STM.

    Args:
        controller: objekt s metodou step(e, dt) -> y (viz controller.py).
        plant: objekt s metodou step(y, dt) -> z_tip a atributem z
            (počáteční poloha), viz plant.py.
        surface_fn: funkce h(x) -> výška povrchu [m].
        v: rychlost pojezdu hrotu [m/s] (záporná = jízda proti ose x).
        kappa: rozpadová konstanta vlnové funkce [1/m].
        V: předpětí [V].
        I_set: požadovaný proud (setpoint) [A].
        g_set: nominální mezera hrot-vzorek při I = I_set [m].
        g_contact: mezera, při které se hlásí náraz [m].
        dt: délka časového kroku, dt << tau [s].
        t_end: konec simulace [s].
        x0: počáteční poloha x tohoto průjezdu [m] (0 = od začátku).
        t0: počáteční čas tohoto průjezdu pro záznam do result.t [s];
            slouží jen k tomu, aby čas v navazujícím (backward) průjezdu
            plynule pokračoval na grafu, do dynamiky nevstupuje.
        current_noise: zdroj šumu měřeného proudu [A] se step(dt), nebo
            None (bez šumu). Přičte se k proudu, ze kterého regulátor
            počítá chybu; skutečný proud se nemění.
        gap_noise: zdroj šumu mezery [m] se step(dt), nebo None. Přičte se
            k fyzické mezeře g (mění proud i detekci nárazu).
            Pro backward průjezd se předávají STEJNÉ instance šumu jako
            forward (šum je funkcí času a musí pokračovat, ne se opakovat).
        callback: volitelná funkce callback(i, stav) volaná po každém kroku,
            nebo None (výchozí, nic se nevolá a chování je beze změny).
            `stav` je slovník s veličinami TOHO kroku (t, x, h, g, I, I_meas,
            e, y, z_tip_novy) - slouží interaktivnímu nástroji ke krokování
            výpočtu před studenty, aby regulátor nebyl černá skříňka.
            Do dynamiky nesmí zasahovat: návratová hodnota se ignoruje
            a smyčka počítá dál ze svých vlastních proměnných.

    Returns:
        SimulationResult se zaznamenanými průběhy a příznakem nárazu
        (I = skutečný proud, I_meas = proud, který vidí regulátor).
    """
    c = c_from_setpoint(V, kappa, g_set, I_set)
    result = SimulationResult()

    n_steps = int(t_end / dt)
    z_tip = plant.z
    for i in range(n_steps):
        t_local = i * dt
        t = t0 + t_local
        x = x0 + v * t_local
        h = surface_fn(x)
        g = z_tip - h
        if gap_noise is not None:
            g += gap_noise.step(dt)
        I = current(V, c, kappa, g)
        if current_noise is None:
            I_meas = I
            e = math.log(I / I_set)
        else:
            # Reálný regulátor měří abs(Current), usměrnění je fyzikální.
            I_meas = I + current_noise.step(dt)
            e = math.log(max(abs(I_meas), I_MIN) / I_set)

        result.t.append(t)
        result.x.append(x)
        result.z_tip.append(z_tip)
        result.g.append(g)
        result.I.append(I)
        result.I_meas.append(I_meas)
        result.e.append(e)

        if g <= g_contact:
            result.crashed = True
            result.crash_index = i
            if callback is not None:
                callback(i, {"t": t, "x": x, "h": h, "g": g, "I": I,
                             "I_meas": I_meas, "e": e, "y": None,
                             "z_tip_novy": z_tip, "crashed": True})
            break

        y = controller.step(e, dt)
        z_tip = plant.step(y, dt)

        if callback is not None:
            callback(i, {"t": t, "x": x, "h": h, "g": g, "I": I,
                         "I_meas": I_meas, "e": e, "y": y,
                         "z_tip_novy": z_tip, "crashed": False})

    return result


@dataclass
class ConstantHeightResult:
    """Výsledek skenu s VYPNUTOU zpětnou vazbou (režim konstantní výšky)."""
    t: list = field(default_factory=list)
    x: list = field(default_factory=list)
    z_tip: list = field(default_factory=list)
    g: list = field(default_factory=list)
    I: list = field(default_factory=list)
    I_meas: list = field(default_factory=list)
    crashed: bool = False
    crash_index: int = None


def constant_height_scan(surface_fn, v, kappa, V, I_set, g_set, g_contact,
                         dt, t_end, z_fixed, x0=0.0, t0=0.0,
                         current_noise=None, gap_noise=None):
    """Sken v konstantní výšce: hrot jede v pevném z, regulátor je vypnutý.

    Protiklad k run_loop(): tady se NIC neintegruje. Hrot jede rychlostí v
    podél x ve stále stejné výšce z_fixed a v každém kroku se jen spočítá,
    jaký proud by na té pozici tekl. Topografie se tím promítne přímo do
    proudu (exponenciálně), místo aby ji smyčka vykompenzovala pohybem
    hrotu.

    K čemu to je: u reálného STM je to rozdíl mezi režimem konstantního
    proudu a konstantní výšky. Ve výuce je to nejnázornější ukázka toho,
    co zpětná vazba vlastně dělá - stejný povrch, stejné parametry, jednou
    se smyčkou a jednou bez ní.

    Kalibrace c je stejná jako v run_loop() (c_from_setpoint), takže obě
    varianty popisují TÝŽ hrot a týž vzorek a jde je vykreslit do
    jednoho grafu.

    Args:
        surface_fn: funkce h(x) -> výška povrchu [m].
        v: rychlost pojezdu hrotu [m/s] (záporná = proti ose x).
        kappa: rozpadová konstanta vlnové funkce [1/m].
        V: předpětí [V].
        I_set: proud, na který je hrot kalibrovaný [A]; tady neslouží jako
            setpoint (není co regulovat), jen ke kalibraci c a jako
            referenční hodnota do grafu.
        g_set: mezera, při které teče I_set [m] (kalibrace c).
        g_contact: mezera, při které se hlásí náraz [m].
        dt: délka časového kroku [s].
        t_end: konec skenu [s].
        z_fixed: pevná výška hrotu [m] - jediný "ovladač" tohoto režimu.
        x0: počáteční poloha x [m].
        t0: počáteční čas pro záznam [s].
        current_noise: zdroj šumu měřeného proudu [A] se step(dt), nebo None.
        gap_noise: zdroj šumu mezery [m] se step(dt), nebo None.

    Returns:
        ConstantHeightResult. Náraz se hlásí stejným kritériem jako
        v run_loop() (g <= g_contact) a sken se v tom bodě zastaví -
        bez regulátoru k němu dojde, jakmile povrch vystoupá nad
        z_fixed - g_contact.
    """
    c = c_from_setpoint(V, kappa, g_set, I_set)
    result = ConstantHeightResult()

    n_steps = int(t_end / dt)
    for i in range(n_steps):
        t_local = i * dt
        t = t0 + t_local
        x = x0 + v * t_local
        h = surface_fn(x)
        g = z_fixed - h
        if gap_noise is not None:
            g += gap_noise.step(dt)
        I = current(V, c, kappa, g)
        if current_noise is None:
            I_meas = I
        else:
            I_meas = I + current_noise.step(dt)

        result.t.append(t)
        result.x.append(x)
        result.z_tip.append(z_fixed)
        result.g.append(g)
        result.I.append(I)
        result.I_meas.append(I_meas)

        if g <= g_contact:
            result.crashed = True
            result.crash_index = i
            break

    return result
