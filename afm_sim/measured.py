"""Parametry FM-AFM smyčky na jednom místě, s uvedením původu.

Obdoba stm_sim/measured.py pro AFM, ale s podstatně jiným stavem poznání:
u STM je kappa ZMĚŘENÁ, tady se kalibrace NEPOVEDLA. Proto je nutné držet
v UI i v závěrech oddělené, co je měření a co ilustrativní odhad:

- ZMĚŘENO: jen šum Δf a šum amplitudy (konstanty SIGMA_DF, F_C_DF,
  SIGMA_AMP, F_C_AMP v afm_sim/noise.py, kalibrováno z naměřených dat).
- ODHAD / typická hodnota: k_cant a f0 (qPlus, tab. 19.1 z literatury).
- NEKALIBROVÁNO: U0, Ra, K_P. Pokus vyfitovat U0/Ra proti naměřené Δf(z)
  skončil NEGATIVNĚ - atomární LJ model neumí vysvětlit, jak pomalu
  naměřená Δf klesá. Hodnoty zůstaly beze změny, jsou ilustrativní.

Z toho plyne pravidlo pro interaktivní nástroj: preset u AFM se nesmí
jmenovat "reálný přístroj" jako u STM. Jmenuje se "typický qPlus" a UI
u něj musí říct, že jde o nekalibrované hodnoty - jinak by nástroj tvrdil
víc, než projekt ví.

Hodnoty musí souhlasit s blokem NASTAVENÍ v run_simulation_afm.py -
hlídá to tests/test_measured.py.
"""

# ------------------- typické hodnoty (literatura, ne měření) ---------------

K_CANT = 1800.0       # N/m, typická tuhost qPlus senzoru
F0 = 30e3             # Hz, rezonanční frekvence
A_OSC = 0.3e-9        # m, amplituda oscilace

# ----------------------- nekalibrované odhady LJ potenciálu ----------------
# Kalibrace proti naměřené Δf(z) neuspěla (poměr délky poklesu ~50x, a i ten
# je nadhodnocený - opravená metodika klade t = 0 jinam). Rozšíření o vdW
# sílu koule a elektrostatiku to zlepšila, ale závěr o pořadí příspěvků
# visí na kinematice couvání, což je HYPOTÉZA, ne zjištěný pohyb hrotu.

U0 = 0.3 * 1.602e-19  # J (~0,3 eV), hloubka LJ jámy - odhad
RA = 0.3e-9           # m, LJ nulový bod - odhad

# ------------------------- nastavení smyčky --------------------------------

D_SET = 1.0e-9        # m, pracovní bod; musí ležet v přitažlivé větvi PŘED
                      # minimem Δf(d) (run_simulation_afm.py to kontroluje
                      # assertem) a nechat rezervu do d_contact na výšku atomu
D_SET_ATOMS_AFM = 0.80e-9  # m, pracovní bod pro řadu BODOVÝCH atomů; 1,0 nm
                           # je tam příliš daleko - rozdíl Δf nad atomem a mezi
                           # atomy by byl ~0,02 Hz, tedy pod šumem 0,08 Hz
TAU = 1e-3            # s, časová konstanta I-složky regulátoru
K_P = 5e-11           # m, NEkalibrováno (srov. STM K_P 0,011 nm)
T_SYS = 100e-6        # s, setrvačnost akčního členu
V_SCAN = 65e-9        # m/s, rychlost pojezdu hrotu

# Šum: konstanty jsou v afm_sim/noise.py i s odvozením. Jediné ZMĚŘENÉ
# hodnoty v tomhle modulu-sousedovi; vstupují přes noise.sum_frekvence()
# a noise.sum_amplitudy().

# -------------------------------- presety ----------------------------------

PRESET_QPLUS = {
    "popis": "Typický qPlus",
    "k_cant": K_CANT,
    "f0": F0,
    "A": A_OSC,
    "U0": U0,
    "Ra": RA,
    "d_set": D_SET,
    "tau": TAU,
    "K_P": K_P,
    "T_SYS": T_SYS,
    "v": V_SCAN,
    "sum_frekvence": True,
    "sum_amplitudy": True,
}

PRESET_IDEAL = {
    "popis": "Učebnicový ideál (bez šumu)",
    "k_cant": K_CANT,
    "f0": F0,
    "A": A_OSC,
    "U0": U0,
    "Ra": RA,
    "d_set": D_SET,
    "tau": TAU,
    "K_P": 0.0,
    "T_SYS": 0.0,
    "v": V_SCAN,
    "sum_frekvence": False,
    "sum_amplitudy": False,
}

PRESETY = {
    "qplus": PRESET_QPLUS,
    "ideal": PRESET_IDEAL,
}
