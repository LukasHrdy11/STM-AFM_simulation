"""Vykreslení výsledků STM simulace - sdílené mezi CLI skriptem a UI.

Proč to nesedí v run_simulation.py: tentýž graf potřebuje jak dávkový
spouštěč (uloží PNG do export/), tak interaktivní panel (kreslí inline do
widgetu). Aby se kreslení nemuselo psát dvakrát, žije tady.

PRAVIDLO: tenhle modul NEZNÁ žádnou cestu. Nesmí v něm být savefig ani
odkaz na export/ - dostane data, vrátí Figure a co se s ní stane, řeší
volající. Jen tak jde kreslení použít i tam, kde se na disk nic neukládá.

Vrácenou figuru je volající povinen po použití zavřít (plt.close), jinak
se při opakovaném překreslování v UI hromadí v paměti.
"""

import matplotlib.pyplot as plt


def _nm(hodnoty):
    """Převod pole metrů na nanometry (kvůli čitelnosti os)."""
    return [h * 1e9 for h in hodnoty]


def fig_prubehy(result_fwd, surface_fn, g_contact, result_bwd=None,
                titulek="", overlay=None, overlay_popis="referenční běh"):
    """Postaví třípanelový graf: výška hrotu, mezera, log-chyba.

    Args:
        result_fwd: SimulationResult forward průjezdu.
        surface_fn: funkce h(x) -> výška povrchu [m] (kreslí se jako
            referenční profil pod dráhu hrotu).
        g_contact: mezera, při které se hlásí náraz [m] (vodorovná čára).
        result_bwd: SimulationResult backward průjezdu, nebo None.
        titulek: dodatek do titulku horního panelu (např. "PI, atoms").
        overlay: starší SimulationResult vykreslený poloprůhledně na
            pozadí, nebo None. Slouží k porovnání dvou běhů v jednom
            grafu (P vs. I vs. PI, šum zapnutý vs. vypnutý, smyčka vs.
            konstantní výška) bez přepínání mezi obrázky.
        overlay_popis: popisek zamknuté křivky v legendě.

    Returns:
        matplotlib Figure (volající ji musí zavřít).
    """
    fig, axes = plt.subplots(3, 1, figsize=(7, 8))

    if overlay is not None:
        _vykresli_overlay(axes, overlay, overlay_popis)

    axes[0].plot(_nm(result_fwd.x), _nm(result_fwd.z_tip), label="z_tip (forward)")
    if result_bwd is not None:
        axes[0].plot(_nm(result_bwd.x), _nm(result_bwd.z_tip),
                     label="z_tip (backward)")
    h_curve = [surface_fn(x) for x in result_fwd.x]
    axes[0].plot(_nm(result_fwd.x), _nm(h_curve), "--", label="h (povrch)")
    axes[0].set_xlabel("x [nm]")
    axes[0].set_ylabel("výška [nm]")
    axes[0].set_title(f"Výška hrotu a povrchu ({titulek})" if titulek
                      else "Výška hrotu a povrchu")
    axes[0].legend()

    axes[1].plot(_nm(result_fwd.x), _nm(result_fwd.g), label="forward")
    if result_bwd is not None:
        axes[1].plot(_nm(result_bwd.x), _nm(result_bwd.g), label="backward")
    axes[1].axhline(g_contact * 1e9, color="red", linestyle=":", label="g_contact")
    axes[1].set_xlabel("x [nm]")
    axes[1].set_ylabel("g [nm]")
    axes[1].set_title("Mezera hrot-vzorek")
    axes[1].legend()

    axes[2].plot(_nm(result_fwd.x), result_fwd.e, label="forward")
    if result_bwd is not None:
        axes[2].plot(_nm(result_bwd.x), result_bwd.e, label="backward")
    axes[2].set_xlabel("x [nm]")
    axes[2].set_ylabel("e [-]")
    axes[2].set_title("Log-chyba regulátoru")
    if result_bwd is not None:
        axes[2].legend()

    fig.tight_layout()
    return fig


def _vykresli_overlay(axes, overlay, popis):
    """Zamknutá referenční křivka na pozadí (šedá, poloprůhledná)."""
    styl = {"color": "gray", "alpha": 0.5, "linestyle": "--", "linewidth": 1.0}
    axes[0].plot(_nm(overlay.x), _nm(overlay.z_tip), label=popis, **styl)
    axes[1].plot(_nm(overlay.x), _nm(overlay.g), label=popis, **styl)
    # Sken v konstantní výšce log-chybu nepočítá (není co regulovat).
    if getattr(overlay, "e", None):
        axes[2].plot(_nm(overlay.x), overlay.e, label=popis, **styl)


def fig_konstantni_vyska(result, surface_fn, I_set, g_contact, titulek="",
                         result_smycka=None):
    """Graf skenu s VYPNUTOU zpětnou vazbou (režim konstantní výšky).

    Ukazuje to, co se u zapnuté smyčky nevidí: topografie se promítne
    rovnou do proudu, místo aby ji regulátor vykompenzoval pohybem hrotu.
    Proud je v logaritmické ose, protože přes hranu roste exponenciálně.

    Args:
        result: ConstantHeightResult z constant_height_scan().
        surface_fn: funkce h(x) -> výška povrchu [m].
        I_set: proud, na který je hrot kalibrovaný [A] (referenční čára).
        g_contact: mezera, při které se hlásí náraz [m].
        titulek: dodatek do titulku horního panelu.
        result_smycka: volitelný SimulationResult TÉHOŽ povrchu se zapnutou
            smyčkou, vykreslený pro přímé srovnání obou režimů.

    Returns:
        matplotlib Figure (volající ji musí zavřít).
    """
    fig, axes = plt.subplots(3, 1, figsize=(7, 8))

    axes[0].plot(_nm(result.x), _nm(result.z_tip),
                 label="z_tip (konstantní výška)")
    if result_smycka is not None:
        axes[0].plot(_nm(result_smycka.x), _nm(result_smycka.z_tip),
                     label="z_tip (se zpětnou vazbou)")
    h_curve = [surface_fn(x) for x in result.x]
    axes[0].plot(_nm(result.x), _nm(h_curve), "--", label="h (povrch)")
    axes[0].set_xlabel("x [nm]")
    axes[0].set_ylabel("výška [nm]")
    axes[0].set_title(f"Hrot stojí, povrch se mění ({titulek})" if titulek
                      else "Hrot stojí, povrch se mění")
    axes[0].legend()

    axes[1].semilogy(_nm(result.x), [I * 1e12 for I in result.I_meas],
                     label="I (bez zpětné vazby)")
    if result_smycka is not None:
        axes[1].semilogy(_nm(result_smycka.x),
                         [I * 1e12 for I in result_smycka.I_meas],
                         label="I (se zpětnou vazbou)")
    axes[1].axhline(I_set * 1e12, color="red", linestyle=":", label="I_set")
    axes[1].set_xlabel("x [nm]")
    axes[1].set_ylabel("I [pA]")
    axes[1].set_title("Proud kopíruje topografii (log. osa)")
    axes[1].legend()

    axes[2].plot(_nm(result.x), _nm(result.g), label="konstantní výška")
    if result_smycka is not None:
        axes[2].plot(_nm(result_smycka.x), _nm(result_smycka.g),
                     label="se zpětnou vazbou")
    axes[2].axhline(g_contact * 1e9, color="red", linestyle=":", label="g_contact")
    axes[2].set_xlabel("x [nm]")
    axes[2].set_ylabel("g [nm]")
    axes[2].set_title("Mezera hrot-vzorek")
    axes[2].legend()

    fig.tight_layout()
    return fig
