"""Vykreslení výsledků FM-AFM simulace - sdílené mezi CLI skriptem a UI.

Obdoba stm_sim/plotting.py (viz tam pro zdůvodnění a pravidla). Platí
totéž: modul NEZNÁ žádnou cestu, nesmí v něm být savefig ani odkaz na
export/ - dostane data, vrátí Figure.

Proti STM přibývá čtvrtý panel (Δf) a jedna zvláštnost: u povrchu
"atoms-AFM" je řada BODOVÝCH atomů, která nemá výšku h(x). Místo profilu
povrchu se proto kreslí značky atomů v rovině z = 0.
"""

import matplotlib.pyplot as plt
import numpy as np


def _nm(hodnoty):
    """Převod pole metrů na nanometry (kvůli čitelnosti os)."""
    return [h * 1e9 for h in hodnoty]


def fig_prubehy(result_fwd, surface_fn, d_contact, df_set, result_bwd=None,
                titulek="", atoms_row=None, overlay=None,
                overlay_popis="referenční běh"):
    """Postaví čtyřpanelový graf: výška hrotu, vzdálenost d, chyba, Δf.

    Args:
        result_fwd: výsledek forward průjezdu (afm_sim.sim.run_loop).
        surface_fn: funkce h(x) -> výška povrchu [m]; ignoruje se, pokud
            je zadaný atoms_row (bodové atomy nemají h(x)).
        d_contact: vzdálenost, při které se hlásí náraz [m].
        df_set: setpoint posunu frekvence [Hz] (referenční čára).
        result_bwd: výsledek backward průjezdu, nebo None.
        titulek: dodatek do titulku horního panelu.
        atoms_row: AtomRow pro povrch "atoms-AFM", nebo None. Když je
            zadaná, kreslí se místo profilu povrchu značky atomů.
        overlay: starší výsledek vykreslený poloprůhledně na pozadí, nebo
            None (porovnání dvou běhů v jednom grafu).
        overlay_popis: popisek zamknuté křivky v legendě.

    Returns:
        matplotlib Figure (volající ji musí zavřít).
    """
    fig, axes = plt.subplots(4, 1, figsize=(7, 10))

    if overlay is not None:
        styl = {"color": "gray", "alpha": 0.5, "linestyle": "--", "linewidth": 1.0}
        axes[0].plot(_nm(overlay.x), _nm(overlay.z_tip), label=overlay_popis, **styl)
        axes[1].plot(_nm(overlay.x), _nm(overlay.d), label=overlay_popis, **styl)
        axes[2].plot(_nm(overlay.x), overlay.e, label=overlay_popis, **styl)
        axes[3].plot(_nm(overlay.x), overlay.df, label=overlay_popis, **styl)

    axes[0].plot(_nm(result_fwd.x), _nm(result_fwd.z_tip), label="z_tip (forward)")
    if result_bwd is not None:
        axes[0].plot(_nm(result_bwd.x), _nm(result_bwd.z_tip),
                     label="z_tip (backward)")
    if atoms_row is not None:
        # Bodové atomy: značky v rovině atomů místo (plochého) h(x).
        axes[0].plot(atoms_row.x * 1e9, np.zeros(atoms_row.n), "ko",
                     label="atomy (body)")
    else:
        h_curve = [surface_fn(x) for x in result_fwd.x]
        axes[0].plot(_nm(result_fwd.x), _nm(h_curve), "--", label="h (povrch)")
    axes[0].set_xlabel("x [nm]")
    axes[0].set_ylabel("výška [nm]")
    axes[0].set_title(f"Výška hrotu a povrchu ({titulek})" if titulek
                      else "Výška hrotu a povrchu")
    axes[0].legend()

    axes[1].plot(_nm(result_fwd.x), _nm(result_fwd.d), label="forward")
    if result_bwd is not None:
        axes[1].plot(_nm(result_bwd.x), _nm(result_bwd.d), label="backward")
    axes[1].axhline(d_contact * 1e9, color="red", linestyle=":", label="d_contact")
    axes[1].set_xlabel("x [nm]")
    axes[1].set_ylabel("d [nm]")
    axes[1].set_title("Vzdálenost hrot-vzorek (mean poloha)")
    axes[1].legend()

    axes[2].plot(_nm(result_fwd.x), result_fwd.e, label="forward")
    if result_bwd is not None:
        axes[2].plot(_nm(result_bwd.x), result_bwd.e, label="backward")
    axes[2].set_xlabel("x [nm]")
    axes[2].set_ylabel("e [Hz]")
    axes[2].set_title("Chyba regulátoru (df_set - Δf)")
    if result_bwd is not None:
        axes[2].legend()

    axes[3].plot(_nm(result_fwd.x), result_fwd.df, label="forward")
    if result_bwd is not None:
        axes[3].plot(_nm(result_bwd.x), result_bwd.df, label="backward")
    axes[3].axhline(df_set, color="gray", linestyle=":", label="df_set")
    axes[3].set_xlabel("x [nm]")
    axes[3].set_ylabel("Δf [Hz]")
    axes[3].set_title("Posun frekvence")
    axes[3].legend()

    fig.tight_layout()
    return fig


def fig_frekvence_kolem_udalosti(result, f0, label, n_okno=300):
    """Zoom rezonanční frekvence f(t) = f0 + Δf(d) kolem nárazu/nestability.

    Stávající data z result.df, jen jiný výřez a osa (čas místo x).

    Fáze se nevykresluje - model je čistě konzervativní (Lennard-Jones),
    bez disipace, takže fázový posun by nebyl reálným výstupem simulace.

    Args:
        result: výsledek s crashed nebo unstable = True.
        f0: rezonanční frekvence cantileveru [Hz].
        label: který průjezd to je ("forward"/"backward"), do popisku osy.
        n_okno: kolik kroků před událostí se vykreslí.

    Returns:
        matplotlib Figure, nebo None, pokud v tomto běhu k události nedošlo.
    """
    if not (result.crashed or result.unstable):
        return None

    i_udalost = result.crash_index if result.crashed else result.unstable_index
    i_start = max(0, i_udalost - n_okno)
    t_okno = [(t - result.t[i_udalost]) * 1e6 for t in result.t[i_start:i_udalost + 1]]
    f_okno = [f0 + df for df in result.df[i_start:i_udalost + 1]]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_okno, [f * 1e-3 for f in f_okno])
    ax.axvline(0.0, color="red", linestyle=":",
               label="NÁRAZ" if result.crashed else "NESTABILITA")
    ax.axhline(f0 * 1e-3, color="gray", linestyle="--", label="f0")
    ax.set_xlabel(f"t - t_událost [µs]  ({label})")
    ax.set_ylabel("f = f0 + Δf [kHz]")
    ax.set_title("Rezonanční frekvence cantileveru kolem nárazu/nestability")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_konstantni_vyska(result, surface_fn, df_set, d_contact, titulek="",
                         atoms_row=None, result_smycka=None):
    """Graf skenu s VYPNUTOU zpětnou vazbou (režim konstantní výšky).

    Obdoba stm_sim.plotting.fig_konstantni_vyska pro AFM. Ukazuje, co se
    se zapnutou smyčkou nevidí: topografie se promítne rovnou do Δf, místo
    aby ji regulátor vykompenzoval pohybem hrotu.

    Δf se na rozdíl od proudu u STM kreslí v lineární ose - není
    exponenciální a navíc je záporná.

    Args:
        result: ConstantHeightResult z afm_sim.sim.constant_height_scan().
        surface_fn: funkce h(x) -> výška povrchu [m].
        df_set: Δf pracovního bodu [Hz] (referenční čára).
        d_contact: vzdálenost, při které se hlásí náraz [m].
        titulek: dodatek do titulku horního panelu.
        atoms_row: AtomRow pro povrch "atoms-AFM", nebo None.
        result_smycka: volitelný výsledek TÉHOŽ povrchu se zapnutou smyčkou,
            vykreslený pro přímé srovnání obou režimů.

    Returns:
        matplotlib Figure (volající ji musí zavřít).
    """
    fig, axes = plt.subplots(3, 1, figsize=(7, 8))

    axes[0].plot(_nm(result.x), _nm(result.z_tip),
                 label="z_tip (konstantní výška)")
    if result_smycka is not None:
        axes[0].plot(_nm(result_smycka.x), _nm(result_smycka.z_tip),
                     label="z_tip (se zpětnou vazbou)")
    if atoms_row is not None:
        axes[0].plot(atoms_row.x * 1e9, np.zeros(atoms_row.n), "ko",
                     label="atomy (body)")
    else:
        h_curve = [surface_fn(x) for x in result.x]
        axes[0].plot(_nm(result.x), _nm(h_curve), "--", label="h (povrch)")
    axes[0].set_xlabel("x [nm]")
    axes[0].set_ylabel("výška [nm]")
    axes[0].set_title(f"Hrot stojí, povrch se mění ({titulek})" if titulek
                      else "Hrot stojí, povrch se mění")
    axes[0].legend()

    axes[1].plot(_nm(result.x), result.df_meas, label="Δf (bez zpětné vazby)")
    if result_smycka is not None:
        axes[1].plot(_nm(result_smycka.x), result_smycka.df_meas,
                     label="Δf (se zpětnou vazbou)")
    axes[1].axhline(df_set, color="red", linestyle=":", label="df_set")
    axes[1].set_xlabel("x [nm]")
    axes[1].set_ylabel("Δf [Hz]")
    axes[1].set_title("Δf kopíruje topografii")
    axes[1].legend()

    axes[2].plot(_nm(result.x), _nm(result.d), label="konstantní výška")
    if result_smycka is not None:
        axes[2].plot(_nm(result_smycka.x), _nm(result_smycka.d),
                     label="se zpětnou vazbou")
    axes[2].axhline(d_contact * 1e9, color="red", linestyle=":", label="d_contact")
    axes[2].set_xlabel("x [nm]")
    axes[2].set_ylabel("d [nm]")
    axes[2].set_title("Vzdálenost hrot-vzorek (mean poloha)")
    axes[2].legend()

    fig.tight_layout()
    return fig
