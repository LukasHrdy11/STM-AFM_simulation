"""Sestavení a spuštění STM simulace z jednoho slovníku parametrů.

Tohle je ta část interaktivního nástroje, která NEZÁVISÍ na ipywidgets:
vezme slovník hodnot (přesně to, co v run_simulation.py dělá blok
NASTAVENÍ ručně) a vrátí hotové výsledky. Widgety v widgets_stm.py pak
jen plní ten slovník.

Rozdělení má dva důvody: jde to testovat bez notebooku (tests/test_interactive.py)
a stejnou funkci může volat i dávkový skript.

Pravidlo přenositelnosti: tenhle modul (a celé interactive/) smí importovat
JEN stm_sim/afm_sim a za běhu nesmí sáhnout na naměřená data v export/,
STM/, literature/ - jinak by nástroj na cizím počítači nefungoval.
Naměřená čísla chodí výhradně přes stm_sim/measured.py.
"""

import numpy as np

from stm_sim import measured, noise
from stm_sim.controller import (
    IntegralController, PIController, ProportionalController, gain_from_tau,
)
from stm_sim.plant import FirstOrderPlant
from stm_sim.sim import constant_height_scan, run_loop
from stm_sim.surface import (
    atoms_surface, generate_mge_steps, make_mackey_glass_surface,
    ramp_surface, step_surface,
)

# Výchozí sada: reálný přístroj + povrch, který se dobře ukazuje (atomy).
# Klíče odpovídají blokům NASTAVENÍ v run_simulation.py.
VYCHOZI = dict(
    measured.PRESET_REALNY,
    controller="PI",
    surface="atoms",
    backward=True,
    sum_proudu_cary=False,   # čáry 312 a 623 Hz k bílému šumu proudu
    zpetna_vazba=True,     # False = režim konstantní výšky (open loop)
    mera_sumu=1.0,         # násobek změřené velikosti šumu
    seed=42,
    # geometrie povrchu
    x_edge=3e-9,
    H=0.2e-9,
    w=0.3e-9,
    mge_pocet=7,
    mge_rozestup=0.6e-9,
    mge_x_step=0.05e-9,
    atoms_start=3e-9,
    atoms_H=0.15e-9,
    atoms_spacing=0.6e-9,
    atoms_sigma=0.10e-9,
    atoms_pocet=6,
)


def sestav_regulator(p):
    """Regulátor podle p['controller'] ("P"/"I"/"PI")."""
    K_I = gain_from_tau(p["kappa"], p["tau"])
    if p["controller"] == "P":
        return ProportionalController(base=p["g_set"], K_P=p["K_P"])
    if p["controller"] == "I":
        return IntegralController(y0=p["g_set"], K_I=K_I)
    if p["controller"] == "PI":
        return PIController(y0=p["g_set"], K_I=K_I, K_P=p["K_P"])
    raise ValueError(f"Neznámý controller: {p['controller']!r} (očekávám P/I/PI)")


def sestav_povrch(p, t_end):
    """Funkce h(x) podle p['surface'] ("step"/"ramp"/"MGE"/"atoms")."""
    if p["surface"] == "step":
        return lambda x: step_surface(x, p["x_edge"], p["H"])
    if p["surface"] == "ramp":
        return lambda x: ramp_surface(x, p["x_edge"], p["H"], p["w"])
    if p["surface"] == "MGE":
        steps = generate_mge_steps(p["mge_pocet"], p["x_edge"],
                                   p["mge_rozestup"], p["H"])
        x_span = max(abs(p["v"]) * t_end, steps[-1][0] + p["mge_x_step"] * 2)
        return make_mackey_glass_surface(steps, x_span, p["mge_x_step"])
    if p["surface"] == "atoms":
        return lambda x: atoms_surface(x, p["atoms_start"], p["atoms_H"],
                                       p["atoms_spacing"], p["atoms_sigma"])
    raise ValueError(f"Neznámý surface: {p['surface']!r} "
                     "(očekávám step/ramp/MGE/atoms)")


def sestav_sum(p):
    """Zdroje šumu (proud, mezera) škálované posuvníkem p['mera_sumu'].

    mera_sumu = 1.0 znamená změřenou velikost (konstanty v stm_sim/noise.py),
    0.0 znamená vypnuto. Škáluje se SIGMA, ne šířka pásma - posuvník mění
    "jak velký" šum je, ne "jaký" šum to je.

    Vrací jedny instance pro forward i backward průjezd; šum je funkcí času
    a musí v backwardu pokračovat, ne se zopakovat.
    """
    rng = np.random.default_rng(p["seed"])
    mira = p["mera_sumu"]

    current_noise = None
    if p["sum_proudu"] and mira > 0:
        bily = noise.LowpassWhiteNoise(rng, noise.SIGMA_PROUD * mira, noise.F_PRE)
        if p.get("sum_proudu_cary", False):
            cary = [(f, a * mira) for f, a in noise.CARY_PROUD]
            current_noise = noise.NoiseSum(bily, noise.SineLines(rng, cary))
        else:
            current_noise = bily

    gap_noise = None
    if p["sum_mezery"] is not None and mira > 0:
        if p["sum_mezery"] == "stojici":
            gap_noise = noise.sum_mezery_stojici(
                rng, asd=noise.ASD_MEZERA_STOJICI * mira)
        elif p["sum_mezery"] == "skeny":
            gap_noise = noise.sum_mezery_skeny(rng)
            if mira != 1.0:
                # PowerLawNoise nemá parametr sigma; škáluje se až výstup.
                gap_noise = _Skalovany(gap_noise, mira)
        else:
            raise ValueError(f"Neznámý sum_mezery: {p['sum_mezery']!r}")

    return current_noise, gap_noise


class _Skalovany:
    """Obal, který vynásobí výstup jiného zdroje šumu konstantou."""

    def __init__(self, zdroj, nasobek):
        self._zdroj = zdroj
        self._nasobek = nasobek

    def step(self, dt):
        return self._zdroj.step(dt) * self._nasobek


def delka_skenu(p):
    """Doba skenu [s] tak, aby hrot projel celý zajímavý úsek povrchu."""
    if p["surface"] == "MGE":
        konec = p["x_edge"] + p["mge_pocet"] * p["mge_rozestup"] + 1e-9
    elif p["surface"] == "atoms":
        konec = p["atoms_start"] + (p["atoms_pocet"] + 0.5) * p["atoms_spacing"]
    else:
        konec = p["x_edge"]
    return 20 * p["tau"] + konec / p["v"]


def spust(parametry=None, callback=None, **zmeny):
    """Spustí simulaci podle parametrů a vrátí výsledky i kontext ke kreslení.

    Args:
        parametry: slovník jako VYCHOZI, nebo None (vezme se VYCHOZI).
        callback: volitelný callback(i, stav) pro krokování (jen forward
            průjezd; backward už krokovat nedává smysl, ten jen dojede).
        **zmeny: jednotlivé klíče k přepsání, ať se nemusí kopírovat
            celý slovník (spust(controller="I") apod.).

    Returns:
        Slovník s klíči: parametry, surface_fn, fwd, bwd (nebo None),
        konstantni_vyska (výsledek open-loop skenu, nebo None), t_end, dt.
        Pro p["zpetna_vazba"] = False se místo smyčky počítá sken
        v konstantní výšce a klíč fwd je None.
    """
    p = dict(VYCHOZI if parametry is None else parametry)
    p.update(zmeny)

    t_end = delka_skenu(p)
    dt = p["tau"] / 100
    surface_fn = sestav_povrch(p, t_end)
    current_noise, gap_noise = sestav_sum(p)

    spolecne = dict(
        surface_fn=surface_fn, kappa=p["kappa"], V=p["V"], I_set=p["I_set"],
        g_set=p["g_set"], g_contact=p["g_contact"], dt=dt, t_end=t_end,
        current_noise=current_noise, gap_noise=gap_noise,
    )

    if not p["zpetna_vazba"]:
        # Režim konstantní výšky: hrot stojí v z_fixed, regulátor nezasahuje.
        # Výchozí výška je pracovní bod smyčky, aby šly oba režimy porovnat.
        z_fixed = p.get("z_fixed", p["g_set"])
        ch = constant_height_scan(v=p["v"], z_fixed=z_fixed, **spolecne)
        return {"parametry": p, "surface_fn": surface_fn, "fwd": None,
                "bwd": None, "konstantni_vyska": ch, "t_end": t_end, "dt": dt}

    controller = sestav_regulator(p)
    plant = FirstOrderPlant(z0=p["g_set"], T_sys=p["T_SYS"])

    fwd = run_loop(controller=controller, plant=plant, v=p["v"],
                   callback=callback, **spolecne)

    bwd = None
    if p["backward"] and not fwd.crashed:
        # Stejné instance regulátoru, akčního členu i šumu - backward
        # navazuje na stav, kde forward skončil (viz run_loop docstring).
        bwd = run_loop(controller=controller, plant=plant, v=-p["v"],
                       x0=fwd.x[-1], t0=fwd.t[-1] + dt, **spolecne)

    return {"parametry": p, "surface_fn": surface_fn, "fwd": fwd, "bwd": bwd,
            "konstantni_vyska": None, "t_end": t_end, "dt": dt}


def posbirej_kroky(parametry=None, pocet=40, **zmeny):
    """Spustí simulaci a vrátí `pocet` rovnoměrně rozložených mezikroků.

    K čemu: krokování výpočtu před studenty. Každý záznam je slovník, jak
    ho dává callback v run_loop() - měřený proud, z něj log-chyba, z ní
    příkaz regulátoru a z něj nová poloha hrotu. Přesně ta posloupnost,
    kterou má demonstrace zviditelnit, aby regulátor nebyl černá skříňka.

    Sbírá se až po doběhnutí (ne za chodu), takže to nezdržuje výpočet
    a posuvník v UI pak jen listuje hotovým seznamem.
    """
    # Odhad délky skenu dopředu, aby se dalo prosívat už v callbacku -
    # jinak by se v paměti nakrátko držely desetitisíce slovníků.
    p = dict(VYCHOZI if parametry is None else parametry)
    p.update(zmeny)
    odhad_kroku = max(1, int(delka_skenu(p) / (p["tau"] / 100)))
    rozestup = max(1, odhad_kroku // pocet)

    zaznamy = []

    def sber(i, stav):
        if i % rozestup == 0 and len(zaznamy) < pocet:
            zaznamy.append(stav)

    vysledek = spust(p, callback=sber)
    return vysledek, zaznamy


def popis_kroku(stav, p):
    """Textový rozpis jednoho kroku smyčky: proud -> chyba -> příkaz -> hrot."""
    radky = [
        f"x = {stav['x'] * 1e9:8.3f} nm      t = {stav['t'] * 1e3:8.3f} ms",
        f"h (povrch)      = {stav['h'] * 1e9:8.4f} nm",
        f"g (mezera)      = {stav['g'] * 1e9:8.4f} nm",
        "",
        f"1) I skutečný   = {stav['I'] * 1e12:8.3f} pA",
        f"   I měřený     = {stav['I_meas'] * 1e12:8.3f} pA   "
        f"(+ šum; regulátor vidí tohle)",
        f"2) e = ln(I/I_set) = {stav['e']:+.5f}        "
        f"(I_set = {p['I_set'] * 1e12:.0f} pA)",
    ]
    if stav["y"] is None:
        radky.append("3) NÁRAZ - smyčka se zastavila, model za nárazem neplatí.")
    else:
        radky += [
            f"3) y = regulátor(e) = {stav['y'] * 1e9:8.4f} nm   (příkaz, ne poloha)",
            f"4) z_hrot = akční člen(y) = {stav['z_tip_novy'] * 1e9:8.4f} nm",
        ]
    return "\n".join(radky)
