"""Korektnostní test: measured.py souhlasí s bloky NASTAVENÍ ve spouštěčích.

Není to pytest test: skript vypíše [OK]/[SELHALO] pro každou kontrolu
a na konci souhrn (stejně jako test_mge.py/test_sum.py).

Proč to existuje: stm_sim/measured.py a afm_sim/measured.py jsou druhá,
komentovaná kopie hodnot z NASTAVENÍ. Duplicita je záměrná - workflow
"parametry se mění editací konstant nahoře ve spouštěči" zůstává, jen se
čísla dají importovat i bez spuštění skriptu (potřebuje to interactive/
a presety v něm). Tenhle test je cena za tu duplicitu: hlídá, aby se obě
kopie nerozešly.

Pokud test selže, NEopravuj ho ztišením - buď se změnilo NASTAVENÍ (pak
změň measured.py a doplň, odkud nová hodnota je), nebo naopak.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_simulation as rs
import run_simulation_afm as rsa
from afm_sim import measured as afm_measured
from stm_sim import measured as stm_measured


def main():
    checks = []

    def check(description, ok, detail):
        checks.append(ok)
        print(f"[{'OK' if ok else 'SELHALO'}] {description}   ({detail})")

    def shoda(nazev, hodnota_nastaveni, hodnota_measured):
        # Přesná rovnost, ne tolerance: jde o dvě kopie TÉHOŽ literálu.
        check(f"{nazev} souhlasí", hodnota_nastaveni == hodnota_measured,
              f"NASTAVENÍ = {hodnota_nastaveni!r}, measured = {hodnota_measured!r}")

    print("--- STM (run_simulation.py vs. stm_sim/measured.py) ---")
    shoda("kappa", rs.kappa, stm_measured.KAPPA)
    shoda("K_P", rs.K_P, stm_measured.K_P)
    shoda("I_set", rs.I_set, stm_measured.I_SET)
    shoda("V", rs.V, stm_measured.V_BIAS)
    shoda("v", rs.v, stm_measured.V_SCAN)
    shoda("tau", rs.tau, stm_measured.TAU)
    shoda("T_SYS", rs.T_SYS, stm_measured.T_SYS)
    shoda("g_set", rs.g_set, stm_measured.G_SET)
    shoda("g_contact", rs.g_contact, stm_measured.G_CONTACT)

    print("\n--- AFM (run_simulation_afm.py vs. afm_sim/measured.py) ---")
    shoda("k_cant", rsa.k_cant, afm_measured.K_CANT)
    shoda("f0", rsa.f0, afm_measured.F0)
    shoda("A", rsa.A, afm_measured.A_OSC)
    shoda("U0", rsa.U0, afm_measured.U0)
    shoda("Ra", rsa.Ra, afm_measured.RA)
    shoda("tau", rsa.tau, afm_measured.TAU)
    shoda("K_P", rsa.K_P, afm_measured.K_P)
    shoda("T_SYS", rsa.T_SYS, afm_measured.T_SYS)
    shoda("v", rsa.v, afm_measured.V_SCAN)
    shoda("ATOMS_AFM_D_SET", rsa.ATOMS_AFM_D_SET, afm_measured.D_SET_ATOMS_AFM)
    # d_set se v run_simulation_afm.py přepisuje pro SURFACE = "atoms-AFM",
    # takže se porovnává jen v ostatních režimech.
    if rsa.SURFACE != "atoms-AFM":
        shoda("d_set", rsa.d_set, afm_measured.D_SET)
    else:
        print("[--] d_set se neporovnává (SURFACE = 'atoms-AFM' ho přepisuje "
              f"na {rsa.d_set!r})")

    print("\n--- presety jsou konzistentní samy se sebou ---")
    realny = stm_measured.PRESET_REALNY
    check("STM preset 'realny' nese změřené kappa a K_P",
          realny["kappa"] == stm_measured.KAPPA and realny["K_P"] == stm_measured.K_P,
          f"kappa = {realny['kappa']:.3e}, K_P = {realny['K_P']:.3e}")
    ideal = stm_measured.PRESET_IDEAL
    check("STM preset 'ideal' má vypnutý šum a T_SYS = 0",
          not ideal["sum_proudu"] and ideal["sum_mezery"] is None
          and ideal["T_SYS"] == 0.0,
          f"sum_proudu = {ideal['sum_proudu']}, sum_mezery = {ideal['sum_mezery']}, "
          f"T_SYS = {ideal['T_SYS']}")
    afm_ideal = afm_measured.PRESET_IDEAL
    check("AFM preset 'ideal' má vypnutý šum Δf i amplitudy",
          not afm_ideal["sum_frekvence"] and not afm_ideal["sum_amplitudy"],
          f"sum_frekvence = {afm_ideal['sum_frekvence']}, "
          f"sum_amplitudy = {afm_ideal['sum_amplitudy']}")
    check("Všechny presety mají klíč 'popis' (UI ho zobrazuje)",
          all("popis" in p for p in list(stm_measured.PRESETY.values())
              + list(afm_measured.PRESETY.values())),
          f"presetů celkem: {len(stm_measured.PRESETY) + len(afm_measured.PRESETY)}")

    print("\nVŠECHNY KONTROLY PROŠLY" if all(checks) else "\nNĚKTERÁ KONTROLA SELHALA")


if __name__ == "__main__":
    main()
