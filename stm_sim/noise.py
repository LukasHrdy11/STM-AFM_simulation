"""Zdroje šumu pro simulační smyčku (šum proudu na I, šum mezery na g).

Čísla a doporučení jsou z export/noise_proud/souhrn.md (šum elektroniky
a šum mezery se stojícím hrotem) a export/noise/souhrn.md (šum mezery ze
skenů). Tady se jen generují.

Každý zdroj je objekt se stavem a metodou step(dt) -> hodnota, stejně jako
akční člen v plant.py. Šum je funkcí ČASU (ne polohy x), takže backward
průjezd se stejnou instancí šumu plynule pokračuje a nezopakuje forward.
Proto se do obou volání run_loop() musí předat STEJNÉ instance.

Vzorky se kvůli rychlosti počítají po blocích a vydávají po jednom; dt
proto musí být během života zdroje stálé.
"""

import math

import numpy as np

# Délka bloku předpočítaných vzorků.
_BLOK = 128


class _BlockNoise:
    """Společný základ: vydává po jednom vzorky z předpočítaného bloku."""

    def __init__(self):
        self._dt = None
        self._buf = []
        self._pos = 0

    def step(self, dt):
        """Vrátí další vzorek šumu (posune čas zdroje o dt)."""
        if self._dt is None:
            self._dt = dt
        elif dt != self._dt:
            raise ValueError(f"Zdroj šumu vyžaduje stálé dt ({self._dt} != {dt})")
        if self._pos >= len(self._buf):
            self._buf = self._block(_BLOK, dt).tolist()
            self._pos = 0
        value = self._buf[self._pos]
        self._pos += 1
        return value

    def samples(self, n, dt):
        """Vrátí n po sobě jdoucích vzorků jako pole (pro testy a analýzy)."""
        return np.array([self.step(dt) for _ in range(n)])

    def _block(self, n, dt):
        raise NotImplementedError


class LowpassWhiteNoise(_BlockNoise):
    """Bílý šum za dolní propustí 1. řádu (proces Ornsteina-Uhlenbecka).

    Přesná diskretizace: b = exp(-2*pi*f_c*dt), x_n = b*x_{n-1} + (1-b)*w_n,
    w ~ N(0, sigma*sqrt((1+b)/(1-b))), takže stacionární směrodatná odchylka
    je přesně sigma. Počáteční stav se losuje ze stacionárního rozdělení
    (žádný náběh). Jednostranná PSD ve spojitém přiblížení:
    S(f) = 2*sigma^2 / (pi*f_c) / (1 + (f/f_c)^2).

    sigma i f_c mohou být pole (několik nezávislých procesů naráz, sečtených
    do jednoho výstupu) - používá PowerLawNoise.
    """

    def __init__(self, rng, sigma, f_c):
        """
        Args:
            rng: numpy Generator (sdílený mezi všemi zdroji a průjezdy).
            sigma: stacionární směrodatná odchylka (skalár nebo pole).
            f_c: rohová frekvence dolní propusti [Hz] (skalár nebo pole).
        """
        super().__init__()
        self.rng = rng
        self.sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
        self.f_c = np.atleast_1d(np.asarray(f_c, dtype=float))
        self._x = self.sigma * rng.standard_normal(self.sigma.shape)

    def _block(self, n, dt):
        b = np.exp(-2.0 * math.pi * self.f_c * dt)[:, None]
        sigma_w = self.sigma[:, None] * np.sqrt((1.0 + b) / (1.0 - b))
        w = sigma_w * self.rng.standard_normal((self.sigma.size, n))
        # x_k = b^k * (x_0 + (1-b) * sum_{j<=k} b^-j * w_j); pro n = 128
        # a f_c <= 50 kHz je b^-n nejvýš ~e^80, takže bez přetečení.
        k = np.arange(1, n + 1)[None, :]
        bk = b ** k
        x = bk * (self._x[:, None] + np.cumsum((1.0 - b) * w / bk, axis=1))
        self._x = x[:, -1]
        return x.sum(axis=0)


class SineLines(_BlockNoise):
    """Součet ostrých čar (sinusovek) s danou rms a náhodnou fází z rng."""

    def __init__(self, rng, lines):
        """
        Args:
            rng: numpy Generator (fáze se losují jednou, tady).
            lines: seznam dvojic (frekvence [Hz], rms).
        """
        super().__init__()
        self.freqs = np.array([f for f, _ in lines], dtype=float)
        self.amps = np.array([rms for _, rms in lines], dtype=float) * math.sqrt(2.0)
        self.phases = rng.uniform(0.0, 2.0 * math.pi, len(lines))
        self._i = 0

    def _block(self, n, dt):
        t = (self._i + np.arange(n)) * dt
        self._i += n
        if self.freqs.size == 0:
            return np.zeros(n)
        arg = 2.0 * math.pi * self.freqs[:, None] * t[None, :] + self.phases[:, None]
        return (self.amps[:, None] * np.sin(arg)).sum(axis=0)


class PowerLawNoise(LowpassWhiteNoise):
    """Šum s ASD ~ f^-alfa (PSD ~ f^-2*alfa) v pásmu F_FIT (5 Hz - 1 kHz).

    Pozor, alfa je exponent ASD - tak ho udává export/noise/souhrn.md
    (fit log ASD v analyza_sumu_stm.py). Normalizuje se na ASD na 10 Hz.

    Součet OU procesů s rohy rozloženými logaritmicky; jejich rozptyly se
    najdou nezáporným fitem na cílovou PSD. Součtem lorentzovských čar
    (každá padá nejvýš jako f^-2) ale nejde dosáhnout PSD strmější než
    ~f^-2, tj. alfa >= 1. Pro alfa >= 1 se proto fituje jen alfa - 1
    a výsledek se protáhne netěsným integrátorem (dolní propust 1. řádu
    s rohem F_INT), který dodá zbylé 1/f v ASD.

    Pod F_FIT se ASD nekontroluje (skeny tam nic neměří) a pomalá složka má
    velký rozptyl (u mediánu ~50 pm). Stav se sice losuje stacionárně, ale
    výstup se posune tak, aby první vzorek byl 0: smyčka před startem
    simulace pomalý drift už dohnala, jinak by na začátku vznikl umělý
    skok mezery. PSD nad ~1/(délka běhu) to nemění.
    """

    F_FIT = (5.0, 1000.0)
    F_INT = 1.0

    def __init__(self, rng, asd_10hz, alfa, f_lo=1.0, f_hi=5000.0, per_decade=4):
        """
        Args:
            rng: numpy Generator.
            asd_10hz: jednostranná ASD na 10 Hz [jednotka/√Hz].
            alfa: exponent ASD (ASD ~ f^-alfa), 0 < alfa < 2.
            f_lo, f_hi: krajní rohové frekvence OU procesů [Hz].
            per_decade: počet rohů na dekádu.
        """
        n = int(round(per_decade * math.log10(f_hi / f_lo))) + 1
        f_k = np.logspace(math.log10(f_lo), math.log10(f_hi), n)
        self.integrate = alfa >= 1.0
        f = np.logspace(math.log10(self.F_FIT[0]), math.log10(self.F_FIT[1]), 80)
        target = (asd_10hz * (10.0 / f) ** alfa) ** 2
        # Matice PSD jednotkových OU procesů (včetně integrátoru), relativně k cíli.
        A = self._psd_unit(f, f_k) / target[:, None]
        var_k = np.full(n, 1.0 / A.sum(axis=1).mean())
        AtA, Atb = A.T @ A, A.T @ np.ones(len(f))
        for _ in range(5000):   # multiplikativní NNLS (Lee-Seung), drží var_k >= 0
            var_k *= Atb / (AtA @ var_k)
        super().__init__(rng, np.sqrt(var_k), f_k)
        self.alfa = alfa
        if self.integrate:
            # Integrátor startuje ze stacionárního rozdělení (bez náběhu).
            fg = np.logspace(-3, 6, 4000)
            self._y = math.sqrt(np.trapezoid(self.psd(fg), fg)) * rng.standard_normal()
        self._offset = None

    def _psd_unit(self, f, f_k):
        """PSD OU procesů s jednotkovým rozptylem (sloupce), spojité přiblížení."""
        f = np.asarray(f, dtype=float)[..., None]
        L = 2.0 / (math.pi * f_k) / (1.0 + (f / f_k) ** 2)
        if self.integrate:
            L = L / (1.0 + (f / self.F_INT) ** 2) * (1.0 + (10.0 / self.F_INT) ** 2)
        return L

    def psd(self, f):
        """Jednostranná PSD tohoto zdroje (spojité přiblížení)."""
        return self._psd_unit(f, self.f_c) @ (self.sigma ** 2)

    def _block(self, n, dt):
        y = self._block_raw(n, dt)
        if self._offset is None:
            self._offset = y[0]
        return y - self._offset

    def _block_raw(self, n, dt):
        u = super()._block(n, dt)
        if not self.integrate:
            return u
        # Netěsný integrátor y_k = b*y_{k-1} + (1-b)*g*u_k, zesílení g drží
        # ASD na 10 Hz (viz _psd_unit); stejný uzavřený tvar jako u OU.
        b = math.exp(-2.0 * math.pi * self.F_INT * dt)
        gain = math.sqrt(1.0 + (10.0 / self.F_INT) ** 2)
        bk = b ** np.arange(1, n + 1)
        y = bk * (self._y + np.cumsum((1.0 - b) * gain * u / bk))
        self._y = y[-1]
        return y


class NoiseSum(_BlockNoise):
    """Součet několika nezávislých zdrojů šumu."""

    def __init__(self, *sources):
        super().__init__()
        self.sources = sources

    def _block(self, n, dt):
        return sum(s._block(n, dt) for s in self.sources)


# ------------------------------- továrny ----------------------------------

# Šum elektroniky (export/noise_proud/souhrn.md, skupina A).
# Kalibrace na sigma bílé části (2,83 pA - změřeno bez předpokladů), ne na
# ASD 0,032 pA/√Hz: ta by s filtrem 1. řádu 7 kHz dala sigma = 3,36 pA, víc
# než změřený CELKOVÝ šum 3,27 pA včetně čar. S 1. řádem pak vychází
# N = sigma / sqrt(pi*f_c/2) = 0,027 pA/√Hz (řádek „1. řád, jeden vzorek").
SIGMA_PROUD = 2.83e-12       # A
F_PRE = 7000.0               # Hz, šířka pásma předzesilovače (údaj obsluhy)
CARY_PROUD = [(312.0, 1.09e-12), (623.0, 0.88e-12)]   # (Hz, A rms) při fs = 4000 Hz

# Šum mezery se stojícím hrotem (skupina B): jen řádový horní odhad.
# Předpoklad: měření sahá do Nyquistu 2 kHz, nad tím se nic neví -> propust 2 kHz.
ASD_MEZERA_STOJICI = 0.07e-12    # m/√Hz (rozsah 0,01-0,07)
F_MEZERA_STOJICI = 2000.0        # Hz

# Šum mezery ze skenů (export/noise/souhrn.md): (ASD na 10 Hz [m/√Hz], alfa ASD).
MEZERA_SKENY = {
    "median": (3.59e-12, 1.08),   # rms 10-500 Hz ~ 10 pm
    "tichy": (0.63e-12, 0.80),    # nejtišší čtvrtina, rms 10-500 Hz ~ 2,5 pm
}


def sum_proudu(rng, cary=True):
    """Šum elektroniky na měřeném proudu I [A]: bílý za propustí 7 kHz (+ čáry)."""
    white = LowpassWhiteNoise(rng, SIGMA_PROUD, F_PRE)
    if not cary:
        return white
    return NoiseSum(white, SineLines(rng, CARY_PROUD))


def sum_mezery_stojici(rng, asd=ASD_MEZERA_STOJICI, f_c=F_MEZERA_STOJICI):
    """Šum mezery g [m] ze stojícího hrotu: bílý s ASD asd za propustí f_c."""
    return LowpassWhiteNoise(rng, asd * math.sqrt(math.pi * f_c / 2.0), f_c)


def sum_mezery_skeny(rng, varianta="median"):
    """Šum mezery g [m] na úrovni skenů (nejhorší varianta): ASD ~ f^-alfa."""
    asd_10hz, alfa = MEZERA_SKENY[varianta]
    return PowerLawNoise(rng, asd_10hz, alfa)
