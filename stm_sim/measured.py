"""Naměřené a zvolené parametry STM smyčky na jednom místě, s uvedením původu.

K čemu to je: interaktivní nástroj (interactive/) a presety v něm potřebují
znát "hodnoty reálného přístroje", ale NESMÍ kvůli tomu číst naměřená data
ani souhrny analýz - ty zůstávají jen v privátním repu (export/, STM/,
noise_files/). Tenhle modul je jediná branka, kterou naměřená čísla do
simulace a do UI vstupují: prosté konstanty + komentář, odkud pocházejí.

Pozor na tři různé kategorie, které se tu potkávají:

- ZMĚŘENO na přístroji: KAPPA (z I(Z)), K_P (z hlavičky .sxm),
  a konstanty šumu (ty žijí v noise.py, viz odkazy níže).
- NASTAVENO při měření: I_SET, V_BIAS, V_SCAN, TAU, T_SYS - hodnoty, se
  kterými se pracovalo, ne výsledky měření.
- ODHAD z literatury: G_SET, G_CONTACT. Model je na nich nezávislý
  v dynamice (chyba e = ln(I/I_set) = -2*kappa*(g - g_set) konstantu c
  vykrátí), takže ABSOLUTNÍ vzdálenosti z tohohle modelu nejsou závěr.

Hodnoty musí souhlasit s blokem NASTAVENÍ v run_simulation.py - hlídá to
tests/test_measured.py, aby se obě kopie nerozešly.
"""

# ----------------------------- změřeno ------------------------------------

# Rozpadová konstanta vlnové funkce [1/m]. Změřeno z I(Z), medián 6 fitů
# (soubory x forward/backward). Nahrazuje dřívější literaturový odhad
# 1,0e10 1/m.
KAPPA = 6.8e9

# Proporcionální zesílení regulátoru [m]. Z hlaviček .sxm: K_p 5,6e-2 m/A
# krát I_set. Rozhodnutí autora věřit hodnotě změřené na přístroji;
# dřívější volba byla 4,5e-9 m. Zdroj rozporu (převod jednotek Nanonisu)
# zůstává nevysvětlen - viz README.
K_P = 0.011e-9

# Šum: konstanty jsou v noise.py (SIGMA_PROUD, F_PRE, CARY_PROUD,
# ASD_MEZERA_STOJICI, F_MEZERA_STOJICI, MEZERA_SKENY) i s odvozením.
# Nekopírovat je sem - vstupují do simulace přes továrny noise.sum_*().

# --------------------------- nastaveno při měření --------------------------

I_SET = 300e-12       # A, setpoint proudu
V_BIAS = 2.5          # V, předpětí
V_SCAN = 65e-9        # m/s, rychlost pojezdu hrotu
TAU = 200e-6          # s, časová konstanta I-složky regulátoru
T_SYS = 100e-6        # s, setrvačnost akčního členu (piezo)

# ----------------------- odhad z literatury (ne závěr) ---------------------

G_SET = 0.6e-9        # m, nominální mezera při I = I_SET
G_CONTACT = 0.25e-9   # m, mezera, při které se hlásí náraz

# -------------------------------- presety ----------------------------------
# Dvě pojmenované sady pro UI: co dělá reálný přístroj vs. čistý princip
# bez šumu. Klíče odpovídají argumentům run_loop() a blokům NASTAVENÍ.

PRESET_REALNY = {
    "popis": "Reálný přístroj (změřené hodnoty)",
    "kappa": KAPPA,
    "K_P": K_P,
    "I_set": I_SET,
    "V": V_BIAS,
    "v": V_SCAN,
    "tau": TAU,
    "T_SYS": T_SYS,
    "g_set": G_SET,
    "g_contact": G_CONTACT,
    "sum_proudu": True,
    "sum_mezery": "stojici",
}

# Učebnicový ideál: kulaté hodnoty, žádný šum, okamžitý akční člen.
# T_SYS = 0 znamená, že P-složka přestane mít vlastní efekt a je jen
# přeškálováním zesílení (viz README, Meze modelu) - pro demonstraci
# samotného principu integrační smyčky je to žádoucí, ne chyba.
PRESET_IDEAL = {
    "popis": "Učebnicový ideál (bez šumu)",
    "kappa": 1.0e10,
    "K_P": 0.0,
    "I_set": 1e-9,
    "V": 1.0,
    "v": 50e-9,
    "tau": 200e-6,
    "T_SYS": 0.0,
    "g_set": 0.5e-9,
    "g_contact": 0.25e-9,
    "sum_proudu": False,
    "sum_mezery": None,
}

PRESETY = {
    "realny": PRESET_REALNY,
    "ideal": PRESET_IDEAL,
}
