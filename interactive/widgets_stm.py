"""Interaktivní STM panel pro dashboard.ipynb (ipywidgets).

Ovladače odpovídají bloku NASTAVENÍ v run_simulation.py - co se tam mění
editací kódu, se tady mění posuvníkem. Výpočet sám je ve vypocet_stm.py,
kreslení v stm_sim/plotting.py; tenhle modul je jen lepidlo mezi nimi.

Tři věci, kvůli kterým panel existuje (a které statický PNG neumí):
- krokování smyčky (proud -> chyba -> příkaz -> poloha hrotu),
- posuvník velikosti šumu až do bodu, kdy regulátor ztrácí stopu povrchu,
- přepínač zpětné vazby (režim konstantního proudu vs. konstantní výšky).

Grafy se renderují staticky do Output widgetu - nepotřebuje to ipympl,
stačí ipywidgets a matplotlib inline.
"""

import ipywidgets as w
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

from stm_sim import measured
from stm_sim.plotting import fig_konstantni_vyska, fig_prubehy

from .vypocet_stm import VYCHOZI, popis_kroku, posbirej_kroky, spust

# Kolik mezikroků se nabídne ke krokování (posuvník "Krok").
POCET_KROKU = 60


def build_stm_panel():
    """Sestaví a vrátí celý STM panel jako jeden widget.

    Returns:
        ipywidgets.Widget k zobrazení (display(build_stm_panel())).
    """
    stav = {"overlay": None, "kroky": [], "posledni": None}

    # ------------------------------ ovladače -------------------------------
    preset = w.Dropdown(
        options=[(p["popis"], klic) for klic, p in measured.PRESETY.items()],
        value="realny", description="Preset:",
        style={"description_width": "100px"})
    regulator = w.Dropdown(options=["P", "I", "PI"], value=VYCHOZI["controller"],
                           description="Regulátor:",
                           style={"description_width": "100px"})
    povrch = w.Dropdown(options=["step", "ramp", "MGE", "atoms"],
                        value=VYCHOZI["surface"], description="Povrch:",
                        style={"description_width": "100px"})

    def posuvnik(popis, hodnota, mini, maxi, krok, jednotka=""):
        return w.FloatSlider(value=hodnota, min=mini, max=maxi, step=krok,
                             description=popis, continuous_update=False,
                             readout_format=".3f",
                             style={"description_width": "100px"},
                             layout=w.Layout(width="360px"))

    t_sys = posuvnik("T_sys [µs]:", VYCHOZI["T_SYS"] * 1e6, 0.0, 500.0, 10.0)
    tau = posuvnik("tau [µs]:", VYCHOZI["tau"] * 1e6, 20.0, 2000.0, 20.0)
    rychlost = posuvnik("v [nm/s]:", VYCHOZI["v"] * 1e9, 5.0, 500.0, 5.0)
    k_p = posuvnik("K_P [nm]:", VYCHOZI["K_P"] * 1e9, 0.0, 1.0, 0.005)
    mira_sumu = posuvnik("Míra šumu [x]:", 1.0, 0.0, 10.0, 0.25)
    # Pevná výška hrotu; má smysl jen s VYPNUTOU zpětnou vazbou. Nízká
    # hodnota je ta zajímavá: ukáže, že bez regulátoru hrot do hrany narazí.
    z_fixed = posuvnik("z_hrot [nm]:", VYCHOZI["g_set"] * 1e9, 0.3, 1.5, 0.01)

    sum_proudu = w.Checkbox(value=VYCHOZI["sum_proudu"], description="šum proudu",
                            indent=False)
    sum_cary = w.Checkbox(value=VYCHOZI["sum_proudu_cary"],
                          description="+ čáry 312/623 Hz", indent=False)
    sum_mezery = w.Dropdown(options=[("bez šumu mezery", None),
                                     ("stojící hrot", "stojici"),
                                     ("ze skenů (nejhorší)", "skeny")],
                            value=VYCHOZI["sum_mezery"], description="Šum mezery:",
                            style={"description_width": "100px"})
    backward = w.Checkbox(value=VYCHOZI["backward"], description="zpětný průjezd",
                          indent=False)
    zpetna_vazba = w.Checkbox(value=True, description="zpětná vazba zapnutá",
                              indent=False)

    prepocitat = w.Button(description="Přepočítat", button_style="primary",
                          icon="refresh")
    zamknout = w.Button(description="Zamknout křivku", icon="lock",
                        tooltip="Uloží aktuální průběh jako šedé pozadí "
                                "pro porovnání s dalším během")
    uvolnit = w.Button(description="Uvolnit", icon="unlock")

    krok = w.IntSlider(value=0, min=0, max=0, description="Krok:",
                       continuous_update=False,
                       style={"description_width": "100px"},
                       layout=w.Layout(width="360px"))
    krokovat = w.Button(description="Načíst kroky", icon="list-ol",
                        tooltip="Spustí simulaci s krokováním a nabídne "
                                "listování mezikroky smyčky")

    graf = w.Output()
    hlaseni = w.HTML()
    vypis_kroku = w.Output()

    # ------------------------------ logika ---------------------------------
    def nacti_preset(_=None):
        """Přepíše ovladače hodnotami presetu.

        Preset MUSÍ nastavit widgety, ne jen slovník: hodnoty se sbírají
        z ovladačů, takže cokoli, co se do widgetu nepromítne, by se hned
        zase přepsalo starou hodnotou a preset by nic nedělal.
        """
        pr = measured.PRESETY[preset.value]
        stav["nacitam"] = True
        try:
            t_sys.value = pr["T_SYS"] * 1e6
            tau.value = pr["tau"] * 1e6
            rychlost.value = pr["v"] * 1e9
            k_p.value = pr["K_P"] * 1e9
            sum_proudu.value = pr["sum_proudu"]
            sum_mezery.value = pr["sum_mezery"]
        finally:
            stav["nacitam"] = False
        prekresli()

    def parametry():
        """Posbírá hodnoty ze všech ovladačů do slovníku pro spust()."""
        p = dict(VYCHOZI)
        # Z presetu se berou jen hodnoty, které nemají vlastní ovladač
        # (kappa, I_set, V, g_set, g_contact); zbytek je na widgetech.
        p.update(measured.PRESETY[preset.value])
        p.update(
            controller=regulator.value,
            surface=povrch.value,
            T_SYS=t_sys.value * 1e-6,
            tau=tau.value * 1e-6,
            v=rychlost.value * 1e-9,
            K_P=k_p.value * 1e-9,
            mera_sumu=mira_sumu.value,
            z_fixed=z_fixed.value * 1e-9,
            sum_proudu=sum_proudu.value,
            sum_proudu_cary=sum_cary.value,
            sum_mezery=sum_mezery.value,
            backward=backward.value,
            zpetna_vazba=zpetna_vazba.value,
        )
        return p

    def shrnuti(vysledek):
        """Jedna věta o tom, jak běh dopadl (náraz / rezerva do nárazu)."""
        p = vysledek["parametry"]
        res = vysledek["fwd"] or vysledek["konstantni_vyska"]
        g_min = min(res.g)
        rezerva = (g_min - p["g_contact"]) * 1e12
        if res.crashed:
            return (f"<b style='color:#b00'>NÁRAZ</b> v x = "
                    f"{res.x[-1] * 1e9:.3f} nm - model za nárazem neplatí.")
        barva = "#b00" if rezerva < 50 else "#060"
        return (f"Bez nárazu. Nejmenší mezera {g_min * 1e9:.4f} nm, "
                f"rezerva do nárazu <b style='color:{barva}'>{rezerva:.0f} pm</b>.")

    def prekresli(_=None):
        if stav.get("nacitam"):
            return   # probíhá načítání presetu, překreslí se až nakonec
        vysledek = spust(parametry())
        stav["posledni"] = vysledek
        p = vysledek["parametry"]
        with graf:
            clear_output(wait=True)
            if vysledek["fwd"] is None:
                # Zamknutá křivka ze smyčky (pozná se podle log-chyby,
                # kterou sken v konstantní výšce nemá) se vykreslí pro
                # přímé srovnání obou režimů nad stejnou hranou.
                zamknuta = stav["overlay"]
                fig = fig_konstantni_vyska(
                    result=vysledek["konstantni_vyska"],
                    surface_fn=vysledek["surface_fn"], I_set=p["I_set"],
                    g_contact=p["g_contact"],
                    titulek=f"{p['surface']}, bez zpětné vazby",
                    result_smycka=(zamknuta if hasattr(zamknuta, "e") else None))
            else:
                fig = fig_prubehy(
                    result_fwd=vysledek["fwd"], surface_fn=vysledek["surface_fn"],
                    g_contact=p["g_contact"], result_bwd=vysledek["bwd"],
                    titulek=f"{p['controller']}, {p['surface']}, "
                            f"T_sys = {p['T_SYS'] * 1e6:.0f} µs",
                    overlay=stav["overlay"])
            display(fig)
            plt.close(fig)
        hlaseni.value = shrnuti(vysledek)

    def zamkni(_):
        if stav["posledni"] is None:
            return
        # Zamkne se to, co je zrovna na obrazovce - se smyčkou i bez ní.
        stav["overlay"] = (stav["posledni"]["fwd"]
                           or stav["posledni"]["konstantni_vyska"])
        prekresli()

    def uvolni(_):
        stav["overlay"] = None
        prekresli()

    def nacti_kroky(_):
        vysledek, kroky = posbirej_kroky(parametry(), pocet=POCET_KROKU)
        stav["kroky"] = kroky
        stav["posledni"] = vysledek
        krok.max = max(0, len(kroky) - 1)
        krok.value = 0
        zobraz_krok()

    def zobraz_krok(_=None):
        with vypis_kroku:
            clear_output(wait=True)
            if not stav["kroky"]:
                print("Nejdřív klikni na „Načíst kroky\".")
                return
            i = min(krok.value, len(stav["kroky"]) - 1)
            p = stav["posledni"]["parametry"]
            print(f"--- krok {i + 1} z {len(stav['kroky'])} "
                  "(rovnoměrně přes celý sken) ---")
            print(popis_kroku(stav["kroky"][i], p))

    # Krokování má smysl jen se zapnutou smyčkou (bez ní regulátor nepočítá).
    def prepni_krokovani(_=None):
        # Krokování má smysl jen se smyčkou, pevná výška naopak jen bez ní.
        krokovat.disabled = not zpetna_vazba.value
        krok.disabled = not zpetna_vazba.value
        z_fixed.disabled = zpetna_vazba.value

    for ovladac in (regulator, povrch, sum_mezery):
        ovladac.observe(prekresli, names="value")
    for ovladac in (t_sys, tau, rychlost, k_p, mira_sumu, z_fixed):
        ovladac.observe(prekresli, names="value")
    preset.observe(nacti_preset, names="value")
    for ovladac in (sum_proudu, sum_cary, backward, zpetna_vazba):
        ovladac.observe(prekresli, names="value")
    zpetna_vazba.observe(prepni_krokovani, names="value")
    prepocitat.on_click(prekresli)
    zamknout.on_click(zamkni)
    uvolnit.on_click(uvolni)
    krokovat.on_click(nacti_kroky)
    krok.observe(zobraz_krok, names="value")

    # ------------------------------ rozvržení ------------------------------
    vlevo = w.VBox([preset, regulator, povrch, sum_mezery,
                    w.HBox([sum_proudu, sum_cary]),
                    w.HBox([backward, zpetna_vazba]),
                    w.HBox([prepocitat, zamknout, uvolnit])],
                   layout=w.Layout(width="440px", flex="0 0 auto"))
    vpravo = w.VBox([t_sys, tau, rychlost, k_p, mira_sumu, z_fixed],
                    layout=w.Layout(width="420px", flex="0 0 auto"))

    krokovani = w.VBox([
        w.HTML("<h4>Krokování smyčky</h4>"
               "<div>Jeden krok simulace: změř proud → spočti log-chybu → "
               "regulátor z ní udělá příkaz → akční člen posune hrot. "
               "Nic jiného smyčka nedělá.</div>"),
        w.HBox([krokovat, krok]),
        vypis_kroku,
    ])

    prepni_krokovani()
    prekresli()

    return w.VBox([
        w.HTML("<h3>STM: zpětná vazba v režimu konstantního proudu</h3>"),
        w.HBox([vlevo, vpravo], layout=w.Layout(justify_content="flex-start")),
        hlaseni,
        graf,
        w.HTML("<hr>"),
        krokovani,
    ])
