"""Korektnostní test interaktivní vrstvy (interactive/).

Není to pytest test: skript vypíše [OK]/[SELHALO] pro každou kontrolu
a na konci souhrn (stejně jako test_mge.py/test_sum.py).

Nejdůležitější kontrola je první: panel musí počítat BIT-PŘESNĚ tutéž
simulaci jako dávkový spouštěč. Kdyby se rozešly, nástroj by studentům
ukazoval něco jiného, než co je v diplomce.

Dál se kontroluje invariant přenositelnosti: interactive/ nesmí sáhnout
na naměřená data (export/, STM/, literature/, ...) ani importovat
analysis/, protože na počítači vyučujícího nic z toho není.
"""

import ast
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")

import numpy as np

KOREN = Path(__file__).resolve().parent.parent

ZAKAZANE_CESTY = ("export/", "STM/", "AFM/", "literature/", "noise_files/")
# Soubory, které smí zakázané řetězce nést v KÓDU (ne jen v komentáři):
# - test_fig_5_11.py si výstupní složku vytváří sám (a vytvoří ji i tam,
#   kde žádná naměřená data nejsou, takže přenositelnost neporušuje),
# - tenhle soubor ty řetězce drží jako DATA kontroly, ne jako cestu.
VYJIMKY_CEST = {"tests/test_fig_5_11.py", "tests/test_interactive.py"}
# Totéž pro import analysis/: test_kalibrace_kappa.py ho potřebuje záměrně
# (proto se nepublikuje), tenhle soubor jen hledá jeho jméno v cizím kódu.
VYJIMKY_ANALYSIS = {"tests/test_kalibrace_kappa.py", "tests/test_interactive.py"}


def _otisk(vysledky, pole):
    h = hashlib.sha256()
    for res in vysledky:
        for nazev in pole:
            h.update(np.asarray(getattr(res, nazev), dtype=np.float64).tobytes())
    return h.hexdigest()


def _kod_bez_komentaru(cesta):
    """Zdroják bez komentářů a docstringů (jen skutečný kód).

    Komentáře o původu konstant (measured.py) cesty zmiňovat MUSÍ; zakázané
    je jen odkazovat se na ně za běhu. Rozdíl se pozná jen po odstranění
    komentářů, grep na to nestačí.
    """
    strom = ast.parse(cesta.read_text())
    for uzel in ast.walk(strom):
        # Docstringy jsou Expr(Constant(str)) - vyhodíme je.
        if isinstance(uzel, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            telo = uzel.body
            if (telo and isinstance(telo[0], ast.Expr)
                    and isinstance(telo[0].value, ast.Constant)
                    and isinstance(telo[0].value.value, str)):
                telo.pop(0)
    return ast.unparse(strom)


def main():
    checks = []

    def check(description, ok, detail):
        checks.append(ok)
        print(f"[{'OK' if ok else 'SELHALO'}] {description}   ({detail})")

    # --- 1. STM: panel vs. run_simulation.py, bit-přesně -------------------
    import run_simulation as rs
    from interactive.vypocet_stm import spust as spust_stm
    from stm_sim.plant import FirstOrderPlant
    from stm_sim.sim import run_loop

    controller = rs.build_controller()
    plant = FirstOrderPlant(z0=rs.g_set, T_sys=rs.T_SYS)
    surface_fn = rs.build_surface()
    cn, gn = rs.build_noise()
    spolecne = dict(surface_fn=surface_fn, kappa=rs.kappa, V=rs.V,
                    I_set=rs.I_set, g_set=rs.g_set, g_contact=rs.g_contact,
                    dt=rs.dt, t_end=rs.t_end, current_noise=cn, gap_noise=gn)
    ref_fwd = run_loop(controller=controller, plant=plant, v=rs.v, **spolecne)
    ref_bwd = run_loop(controller=controller, plant=plant, v=-rs.v,
                       x0=ref_fwd.x[-1], t0=ref_fwd.t[-1] + rs.dt, **spolecne)

    panel = spust_stm()
    pole_stm = ("t", "x", "z_tip", "g", "I", "I_meas", "e")
    check("STM panel počítá bit-přesně totéž co run_simulation.py",
          _otisk([panel["fwd"], panel["bwd"]], pole_stm)
          == _otisk([ref_fwd, ref_bwd], pole_stm),
          f"kroků forward = {len(ref_fwd.t)}")

    # --- 2. AFM: panel vs. run_simulation_afm.py --------------------------
    import run_simulation_afm as rsa
    from afm_sim.frequency_shift import frequency_shift
    from afm_sim.sim import run_loop as run_loop_afm
    from interactive.vypocet_afm import spust as spust_afm

    df_set = frequency_shift(rsa.d_set, rsa.A, rsa.k_cant, rsa.f0, rsa.force_ref)
    ctrl_a = rsa.build_controller()
    plant_a = FirstOrderPlant(z0=rsa.d_set, T_sys=rsa.T_SYS)
    fn, gn_a = rsa.build_noise()
    spolecne_a = dict(
        surface_fn=rsa.build_surface(), force_fn=rsa.force_fn, A=rsa.A,
        k_cant=rsa.k_cant, f0=rsa.f0, df_set=df_set, d_contact=rsa.d_contact,
        dt=rsa.dt, t_end=rsa.t_end,
        adaptive_tau=rsa.tau if rsa.GAIN_MODE == "local" else None,
        freq_noise=fn, gap_noise=gn_a, force_fn_xz=rsa.force_fn_xz,
        amp_noise=rsa.build_amp_noise())
    ref_a_fwd = run_loop_afm(controller=ctrl_a, plant=plant_a, v=rsa.v,
                             x0=rsa.X0_SKEN, **spolecne_a)
    ref_a_bwd = run_loop_afm(controller=ctrl_a, plant=plant_a, v=-rsa.v,
                             x0=ref_a_fwd.x[-1], t0=ref_a_fwd.t[-1] + rsa.dt,
                             **spolecne_a)

    panel_a = spust_afm()
    pole_afm = ("t", "x", "z_tip", "d", "e", "df")
    check("AFM panel počítá bit-přesně totéž co run_simulation_afm.py",
          _otisk([panel_a["fwd"], panel_a["bwd"]], pole_afm)
          == _otisk([ref_a_fwd, ref_a_bwd], pole_afm),
          f"kroků forward = {len(ref_a_fwd.t)}")

    # --- 3. posuvník šumu opravdu škáluje šum ------------------------------
    bez = spust_stm(mera_sumu=0.0)
    slabe = spust_stm(mera_sumu=1.0)
    silne = spust_stm(mera_sumu=5.0)

    def rozptyl(v):
        g = np.asarray(v["fwd"].g)
        return float(np.std(g - np.mean(g)))

    check("Míra šumu 0 znamená opravdu žádný šum (I_meas == I)",
          all(a == b for a, b in zip(bez["fwd"].I, bez["fwd"].I_meas)),
          f"kroků = {len(bez['fwd'].I)}")
    check("Větší míra šumu = větší rozkmit mezery",
          rozptyl(bez) < rozptyl(slabe) < rozptyl(silne),
          f"std(g): {rozptyl(bez) * 1e12:.2f} < {rozptyl(slabe) * 1e12:.2f} "
          f"< {rozptyl(silne) * 1e12:.2f} pm")

    # --- 4. režim bez zpětné vazby -----------------------------------------
    open_loop = spust_stm(zpetna_vazba=False)
    check("Vypnutá zpětná vazba vrací sken v konstantní výšce",
          open_loop["fwd"] is None and open_loop["konstantni_vyska"] is not None,
          f"kroků = {len(open_loop['konstantni_vyska'].t)}")

    # --- 5. krokování ------------------------------------------------------
    from interactive.vypocet_stm import popis_kroku, posbirej_kroky
    vysledek, kroky = posbirej_kroky(pocet=10)
    check("Krokování vrátí požadovaný počet mezikroků",
          len(kroky) == 10, f"kroků = {len(kroky)}")
    text = popis_kroku(kroky[5], vysledek["parametry"])
    check("Rozpis kroku obsahuje celou posloupnost proud -> chyba -> příkaz -> hrot",
          all(klic in text for klic in ("I měřený", "e = ln(I/I_set)",
                                        "regulátor(e)", "akční člen")),
          f"{len(text.splitlines())} řádků")

    # --- 6. panely se sestaví (bez kernelu, jen konstrukce) ----------------
    from interactive.vypocet_stm import VYCHOZI as VYCHOZI_STM
    from interactive.widgets_afm import build_afm_panel
    from interactive.widgets_stm import build_stm_panel
    check("STM panel se sestaví", len(build_stm_panel().children) > 0, "VBox")
    check("AFM panel se sestaví", len(build_afm_panel().children) > 0, "VBox")

    # --- 6b. preset se OPRAVDU promítne do ovladačů ------------------------
    # Tohle je past, do které se dá snadno spadnout: preset se zamíchá do
    # slovníku, ale widgety ho hned zase přepíšou vlastními hodnotami a
    # přepnutí presetu pak nedělá skoro nic. Kontrola dicts v
    # test_measured.py to nechytí - musí se sáhnout na sestavený panel.
    import ipywidgets as ipw

    def najdi(korenovy, typ, popis):
        nalezene = []

        def projdi(widget):
            if isinstance(widget, typ) and getattr(widget, "description", "") == popis:
                nalezene.append(widget)
            for dite in getattr(widget, "children", ()):
                projdi(dite)

        projdi(korenovy)
        return nalezene[0] if nalezene else None

    stm_panel = build_stm_panel()
    preset_stm = najdi(stm_panel, ipw.Dropdown, "Preset:")
    t_sys_stm = najdi(stm_panel, ipw.FloatSlider, "T_sys [µs]:")
    sum_stm = najdi(stm_panel, ipw.Checkbox, "šum proudu")
    preset_stm.value = "ideal"
    check("STM preset 'ideal' opravdu vypne šum a nastaví T_sys = 0",
          t_sys_stm.value == 0.0 and sum_stm.value is False,
          f"T_sys = {t_sys_stm.value} µs, šum proudu = {sum_stm.value}")
    preset_stm.value = "realny"
    check("STM preset 'realny' vrátí změřené hodnoty",
          t_sys_stm.value == VYCHOZI_STM["T_SYS"] * 1e6 and sum_stm.value is True,
          f"T_sys = {t_sys_stm.value} µs, šum proudu = {sum_stm.value}")

    afm_panel = build_afm_panel()
    preset_afm = najdi(afm_panel, ipw.Dropdown, "Preset:")
    sum_df = najdi(afm_panel, ipw.Checkbox, "šum Δf")
    t_sys_afm = najdi(afm_panel, ipw.FloatSlider, "T_sys [µs]:")
    preset_afm.value = "ideal"
    check("AFM preset 'ideal' opravdu vypne šum Δf a nastaví T_sys = 0",
          sum_df.value is False and t_sys_afm.value == 0.0,
          f"šum Δf = {sum_df.value}, T_sys = {t_sys_afm.value} µs")

    # --- 6c. pevná výška jde nastavit z panelu a umí vyrobit náraz --------
    # Bez vlastního ovladače by open-loop demonstrace nikdy nenarazila -
    # výchozí z_fixed je nad povrchem a náraz je přesně to, co má ukázat.
    z_fixed_w = najdi(stm_panel, ipw.FloatSlider, "z_hrot [nm]:")
    check("STM panel má posuvník pevné výšky hrotu", z_fixed_w is not None,
          f"rozsah {z_fixed_w.min}-{z_fixed_w.max} nm" if z_fixed_w else "chybí")
    naraz = spust_stm(zpetna_vazba=False, surface="step",
                      z_fixed=0.30e-9, H=0.2e-9)
    check("Nízká pevná výška z rozsahu posuvníku vyrobí náraz",
          naraz["konstantni_vyska"].crashed
          and z_fixed_w.min <= 0.30 <= z_fixed_w.max,
          f"náraz v x = {naraz['konstantni_vyska'].x[-1] * 1e9:.3f} nm")

    # --- 7. rezonanční křivka je Lorentzova --------------------------------
    from interactive.vypocet_afm import VYCHOZI as VYCHOZI_AFM
    from interactive.vypocet_afm import rezonancni_krivka
    p_afm = dict(VYCHOZI_AFM, Q=5000.0)
    posun = -5.0
    f, amp, faze = rezonancni_krivka(p_afm, df=posun)
    f_max = float(f[int(np.argmax(amp))])
    check("Vrchol rezonance leží na f0 + Δf",
          abs(f_max - (p_afm["f0"] + posun)) < 1.0,
          f"vrchol {f_max:.2f} Hz, očekáváno {p_afm['f0'] + posun:.2f} Hz")
    check("Amplituda ve vrcholu odpovídá Q",
          abs(float(amp.max()) - p_afm["Q"]) / p_afm["Q"] < 0.01,
          f"amplituda = {float(amp.max()):.1f}, Q = {p_afm['Q']:.0f}")
    check("Fáze prochází -90° v rezonanci",
          abs(float(faze[int(np.argmax(amp))]) + 90.0) < 1.0,
          f"fáze ve vrcholu = {float(faze[int(np.argmax(amp))]):.2f}°")

    # --- 8. invariant přenositelnosti --------------------------------------
    publikovane = []
    for slozka in ("interactive", "stm_sim", "afm_sim", "tests"):
        publikovane += sorted((KOREN / slozka).glob("*.py"))

    hrisnici = []
    for cesta in publikovane:
        rel = cesta.relative_to(KOREN).as_posix()
        if rel in VYJIMKY_CEST:
            continue
        kod = _kod_bez_komentaru(cesta)
        for zakazana in ZAKAZANE_CESTY:
            if zakazana in kod:
                hrisnici.append(f"{rel}: {zakazana}")
    check("Žádný publikovaný modul nesahá v KÓDU na naměřená data",
          not hrisnici, "; ".join(hrisnici) if hrisnici
          else f"prověřeno {len(publikovane)} souborů")

    importy_analysis = []
    for cesta in publikovane:
        rel = cesta.relative_to(KOREN).as_posix()
        if rel in VYJIMKY_ANALYSIS:
            continue
        zdroj = cesta.read_text()
        if "from analysis" in zdroj or "import analysis" in zdroj:
            importy_analysis.append(rel)
    check("Žádný publikovaný modul neimportuje analysis/",
          not importy_analysis, "; ".join(importy_analysis)
          if importy_analysis else "prověřeno")

    kreslici = [KOREN / "stm_sim" / "plotting.py", KOREN / "afm_sim" / "plotting.py"]
    s_cestou = [c.name for c in kreslici
                if "savefig" in _kod_bez_komentaru(c)
                or "EXPORT" in _kod_bez_komentaru(c)]
    check("Kreslicí moduly neznají žádnou cestu (bez savefig a EXPORT_DIR)",
          not s_cestou, "; ".join(s_cestou) if s_cestou else "plotting.py x2")

    print("\nVŠECHNY KONTROLY PROŠLY" if all(checks) else "\nNĚKTERÁ KONTROLA SELHALA")


if __name__ == "__main__":
    main()
