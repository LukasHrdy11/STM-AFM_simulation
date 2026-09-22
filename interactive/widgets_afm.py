"""Interaktivní FM-AFM panel pro dashboard.ipynb (ipywidgets).

Obdoba widgets_stm.py (viz tam pro rozdělení výpočet/widgety). Navíc má
panel rezonanční křivky, který ukazuje, co Δf vlastně je: posun CELÉ
rezonance, ne přímo měřená veličina.

Preset se jmenuje "typický qPlus", ne "reálný přístroj": kalibrace
parametrů hrotu se nepovedla a změřený je tu jen šum (viz
afm_sim/measured.py, kde je u každé konstanty uvedeno, odkud je).
"""

import ipywidgets as w
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

from afm_sim import measured
from afm_sim.plotting import fig_frekvence_kolem_udalosti, fig_prubehy

from .vypocet_afm import VYCHOZI, rezonancni_krivka, spust


def build_afm_panel():
    """Sestaví a vrátí celý FM-AFM panel jako jeden widget."""
    stav = {"overlay": None, "posledni": None}

    preset = w.Dropdown(
        options=[(p["popis"], klic) for klic, p in measured.PRESETY.items()],
        value="qplus", description="Preset:",
        style={"description_width": "100px"})
    regulator = w.Dropdown(options=["P", "I", "PI"], value=VYCHOZI["controller"],
                           description="Regulátor:",
                           style={"description_width": "100px"})
    povrch = w.Dropdown(options=["step", "ramp", "atoms", "atoms-AFM"],
                        value=VYCHOZI["surface"], description="Povrch:",
                        style={"description_width": "100px"})
    gain_mode = w.Dropdown(
        options=[("fixed (z pracovního bodu)", "fixed"),
                 ("local (přepočítává se)", "local")],
        value=VYCHOZI["gain_mode"], description="Citlivost:",
        style={"description_width": "100px"}, layout=w.Layout(width="360px"))

    def posuvnik(popis, hodnota, mini, maxi, krok, format=".3f"):
        return w.FloatSlider(value=hodnota, min=mini, max=maxi, step=krok,
                             description=popis, continuous_update=False,
                             readout_format=format,
                             style={"description_width": "100px"},
                             layout=w.Layout(width="340px"))

    d_set = posuvnik("d_set [nm]:", VYCHOZI["d_set"] * 1e9, 0.5, 2.0, 0.02)
    amplituda = posuvnik("A [nm]:", VYCHOZI["A"] * 1e9, 0.05, 1.5, 0.05)
    tau = posuvnik("tau [ms]:", VYCHOZI["tau"] * 1e3, 0.1, 10.0, 0.1)
    t_sys = posuvnik("T_sys [µs]:", VYCHOZI["T_SYS"] * 1e6, 0.0, 500.0, 10.0)
    rychlost = posuvnik("v [nm/s]:", VYCHOZI["v"] * 1e9, 5.0, 500.0, 5.0)
    mira_sumu = posuvnik("Míra šumu [x]:", 1.0, 0.0, 10.0, 0.25)

    sum_frekvence = w.Checkbox(value=VYCHOZI["sum_frekvence"],
                               description="šum Δf", indent=False)
    sum_amplitudy = w.Checkbox(value=VYCHOZI["sum_amplitudy"],
                               description="šum amplitudy", indent=False)
    backward = w.Checkbox(value=VYCHOZI["backward"],
                          description="zpětný průjezd", indent=False)

    prepocitat = w.Button(description="Přepočítat", button_style="primary",
                          icon="refresh")
    zamknout = w.Button(description="Zamknout křivku", icon="lock")
    uvolnit = w.Button(description="Uvolnit", icon="unlock")

    graf = w.Output()
    graf_udalost = w.Output()
    hlaseni = w.HTML()

    # --- rezonanční křivka (vlastní podpanel) ------------------------------
    q_faktor = w.FloatLogSlider(value=5000.0, base=10, min=2, max=5.5,
                                step=0.1, description="Q:",
                                continuous_update=False,
                                style={"description_width": "100px"},
                                layout=w.Layout(width="340px"))
    df_posun = posuvnik("Δf [Hz]:", -5.0, -50.0, 5.0, 0.5, format=".1f")
    graf_rezonance = w.Output()

    def nacti_preset(_=None):
        """Přepíše ovladače hodnotami presetu.

        Preset MUSÍ nastavit widgety, ne jen slovník - hodnoty se sbírají
        z ovladačů, takže co se do widgetu nepromítne, se hned zase
        přepíše zpátky a preset by nedělal nic.
        """
        pr = measured.PRESETY[preset.value]
        stav["nacitam"] = True
        try:
            d_set.value = pr["d_set"] * 1e9
            amplituda.value = pr["A"] * 1e9
            tau.value = pr["tau"] * 1e3
            t_sys.value = pr["T_SYS"] * 1e6
            rychlost.value = pr["v"] * 1e9
            sum_frekvence.value = pr["sum_frekvence"]
            sum_amplitudy.value = pr["sum_amplitudy"]
        finally:
            stav["nacitam"] = False
        prekresli()

    def parametry():
        p = dict(VYCHOZI)
        # Z presetu se berou jen hodnoty bez vlastního ovladače (U0, Ra,
        # k_cant, f0); zbytek nastavuje nacti_preset() do widgetů.
        p.update(measured.PRESETY[preset.value])
        p.update(
            controller=regulator.value,
            surface=povrch.value,
            gain_mode=gain_mode.value,
            d_set=d_set.value * 1e-9,
            A=amplituda.value * 1e-9,
            tau=tau.value * 1e-3,
            T_SYS=t_sys.value * 1e-6,
            v=rychlost.value * 1e-9,
            mera_sumu=mira_sumu.value,
            sum_frekvence=sum_frekvence.value,
            sum_amplitudy=sum_amplitudy.value,
            backward=backward.value,
            Q=q_faktor.value,
        )
        return p

    def shrnuti(vysledek):
        res = vysledek["fwd"]
        d_min = min(res.d)
        rezerva = (d_min - vysledek["d_contact"]) * 1e12
        if res.crashed:
            return (f"<b style='color:#b00'>NÁRAZ</b> v x = "
                    f"{res.x[-1] * 1e9:.3f} nm.")
        if res.unstable:
            return ("<b style='color:#b00'>ZTRÁTA STABILITY</b> - hrot se "
                    "dostal za minimum Δf(d), citlivost změnila znaménko "
                    "a smyčka reguluje opačně. U režimu „local\" je to "
                    "očekávaný jev, ne chyba.")
        barva = "#b00" if rezerva < 50 else "#060"
        return (f"Bez nárazu i nestability. Nejmenší d = {d_min * 1e9:.4f} nm, "
                f"rezerva <b style='color:{barva}'>{rezerva:.0f} pm</b>, "
                f"Δf setpoint = {vysledek['df_set']:.3f} Hz.")

    def prekresli(_=None):
        if stav.get("nacitam"):
            return   # probíhá načítání presetu, překreslí se až nakonec
        p = parametry()
        with graf:
            clear_output(wait=True)
            try:
                vysledek = spust(p)
            except ValueError as chyba:
                # Starý graf se nesmí nechat na obrazovce - patřil jinému
                # nastavení a u chybové hlášky by mátl.
                hlaseni.value = f"<b style='color:#b00'>{chyba}</b>"
                stav["posledni"] = None
                print("Graf se nepřekreslil, oprav nastavení výše.")
                return
            stav["posledni"] = vysledek
            fig = fig_prubehy(
                result_fwd=vysledek["fwd"], surface_fn=vysledek["surface_fn"],
                d_contact=vysledek["d_contact"], df_set=vysledek["df_set"],
                result_bwd=vysledek["bwd"], atoms_row=vysledek["atoms_row"],
                titulek=f"{p['controller']}, {p['surface']}, "
                        f"citlivost {p['gain_mode']}",
                overlay=stav["overlay"])
            display(fig)
            plt.close(fig)
        hlaseni.value = shrnuti(stav["posledni"])

        # Zoom frekvence kolem události se kreslí, jen když k ní došlo.
        with graf_udalost:
            clear_output(wait=True)
            vysledek = stav["posledni"]
            for res, label in ((vysledek["fwd"], "forward"),
                               (vysledek["bwd"], "backward")):
                if res is None:
                    continue
                fig_u = fig_frekvence_kolem_udalosti(res, p["f0"], label)
                if fig_u is not None:
                    display(fig_u)
                    plt.close(fig_u)
                    break

    def prekresli_rezonanci(_=None):
        p = parametry()
        f, amp, faze = rezonancni_krivka(p, df=0.0)
        f_p, amp_p, faze_p = rezonancni_krivka(p, df=df_posun.value)
        with graf_rezonance:
            clear_output(wait=True)
            fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
            axes[0].plot(f * 1e-3, amp, "--", color="gray",
                         label="volný cantilever (Δf = 0)")
            axes[0].plot(f_p * 1e-3, amp_p,
                         label=f"u vzorku (Δf = {df_posun.value:.1f} Hz)")
            axes[0].set_ylabel("amplituda [rel.]")
            axes[0].set_title("Rezonanční křivka: Δf je posun CELÉ rezonance")
            axes[0].legend()
            axes[1].plot(f * 1e-3, faze, "--", color="gray")
            axes[1].plot(f_p * 1e-3, faze_p)
            axes[1].set_xlabel("f [kHz]")
            axes[1].set_ylabel("fáze [°]")
            fig.tight_layout()
            display(fig)
            plt.close(fig)

    def zamkni(_):
        if stav["posledni"] is not None:
            stav["overlay"] = stav["posledni"]["fwd"]
            prekresli()

    def uvolni(_):
        stav["overlay"] = None
        prekresli()

    for ovladac in (regulator, povrch, gain_mode, d_set, amplituda,
                    tau, t_sys, rychlost, mira_sumu, sum_frekvence,
                    sum_amplitudy, backward):
        ovladac.observe(prekresli, names="value")
    preset.observe(nacti_preset, names="value")
    for ovladac in (q_faktor, df_posun, amplituda):
        ovladac.observe(prekresli_rezonanci, names="value")
    prepocitat.on_click(prekresli)
    zamknout.on_click(zamkni)
    uvolnit.on_click(uvolni)

    vlevo = w.VBox([preset, regulator, povrch, gain_mode,
                    w.HBox([sum_frekvence, sum_amplitudy, backward]),
                    w.HBox([prepocitat, zamknout, uvolnit])],
                   layout=w.Layout(width="440px", flex="0 0 auto"))
    vpravo = w.VBox([d_set, amplituda, tau, t_sys, rychlost, mira_sumu],
                    layout=w.Layout(width="420px", flex="0 0 auto"))

    def prepni_d_set(_=None):
        # U řady BODOVÝCH atomů si pracovní bod určuje vypocet_afm sám
        # (d_set z presetu je tam příliš daleko), posuvník by jen mátl.
        je_atoms_afm = povrch.value == "atoms-AFM"
        d_set.disabled = je_atoms_afm
        d_set.description = ("d_set (pevné) [nm]:" if je_atoms_afm
                             else "d_set [nm]:")

    povrch.observe(prepni_d_set, names="value")
    prepni_d_set()
    prekresli()
    prekresli_rezonanci()

    return w.VBox([
        w.HTML("<h3>FM-AFM: zpětná vazba na konstantní Δf</h3>"),
        w.HBox([vlevo, vpravo], layout=w.Layout(justify_content="flex-start")),
        hlaseni,
        graf,
        graf_udalost,
        w.HTML("<hr><h4>Co je vlastně Δf</h4>"
               "<div>Δf není měřená veličina - je to posun rezonance "
               "cantileveru, když na hrot začne působit síla od vzorku. "
               "Posuvníkem Δf se dá vidět, o kolik se celá křivka posune.</div>"),
        w.HBox([q_faktor, df_posun]),
        graf_rezonance,
    ])
