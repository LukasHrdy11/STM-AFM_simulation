"""Zdroj šumu pro FM-AFM smyčku (šum měřeného Δf).

Číslo je z export/sum_frekvence_afm/souhrn.md (vakuové části couvání
z kontaktu, 14. 9. 2026). Generátor (`LowpassWhiteNoise`) se přebírá ze
`stm_sim.noise` beze změny - je fyzikálně neutrální (pracuje jen s `dt`,
`sigma`, `f_c`), stejný precedens jednosměrné závislosti `afm_sim` ->
`stm_sim` už existuje přes `stm_sim.controller`/`plant`/`surface`
v run_simulation_afm.py.

Rozhraní a pravidla použití (sdílený rng, stejné instance pro forward
i backward průjezd) jsou stejná jako u stm_sim/noise.py - viz tam.

Šum mezery (`gap_noise` v afm_sim/sim.py) pro AFM zůstává NEIMPLEMENTOVANÝ -
dostupná data (stojící/couvající hrot) ho neměří, viz Meze v
export/sum_frekvence_afm/souhrn.md.
"""

from stm_sim.noise import LowpassWhiteNoise

# Šum měřeného Δf (export/sum_frekvence_afm/souhrn.md): kalibrováno na
# změřenou sigma po odečtu reziduálního trendu (ne na ASD - stejná filozofie
# jako SIGMA_PROUD u STM). F_C_DF je horní odhad korelačního času (jistota
# jen fs/2 = 25 Hz, nad čím dataset nic neříká - viz Meze v souhrn.md).
SIGMA_DF = 0.0814       # Hz
F_C_DF = 25.0           # Hz

# Šum amplitudy oscilace (kanál OC D1 Amplitude, export/sum_frekvence_afm/souhrn.md): std
# 1,91 a 1,82 pm na segmentech couvání při A = 0,9 nm, průměr 1,865 pm. Bere se jako ABSOLUTNÍ
# hodnota (stejná filozofie jako SIGMA_DF, nepřepočítává se na amplitudu simulace) a se stejným
# předpokladem korelačního času jako Δf (jistota jen fs/2 = 25 Hz). Pro nezávislost na šumu Δf
# se má losovat z SAMOSTATNÉHO generátoru (viz run_simulation_afm.py::build_amp_noise).
SIGMA_AMP = 1.865e-12   # m
F_C_AMP = 25.0          # Hz


def sum_frekvence(rng, sigma=SIGMA_DF, f_c=F_C_DF):
    """Šum měřeného Δf [Hz]: bílý za dolní propustí f_c."""
    return LowpassWhiteNoise(rng, sigma, f_c)


def sum_amplitudy(rng, sigma=SIGMA_AMP, f_c=F_C_AMP):
    """Šum amplitudy oscilace [m]: bílý za dolní propustí f_c. Přičítá se k A v každém kroku
    (`run_loop(..., amp_noise=...)`), takže se do Δf dostane přes závislost Δf(A) (rov. 17.15)."""
    return LowpassWhiteNoise(rng, sigma, f_c)
