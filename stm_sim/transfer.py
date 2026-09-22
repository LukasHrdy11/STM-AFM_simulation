"""Linearizovaný diskrétní přenos smyčky a odhad spektra (pro práci se šumem).

Přenos odpovídá PŘESNĚ rekurencím v controller.py, plant.py a sim.py
(explicitní Euler s krokem dt), ne spojitému modelu. Na tom záleží:
spojitá PI smyčka s akčním členem 1. řádu je stabilní pro libovolné K_P,
kdežto diskrétní má mez K_P, danou krokem dt (nestabilní pól na z = -1,
tj. na Nyquistově frekvenci simulace - artefakt diskretizace, ne fyzika).

Linearizace kolem g_set (rovný povrch):
    e = -2*kappa*(g - g_set) + dI/I_set,   g = z_tip - h + dg
    L(z) = 2*kappa * C(z) * P(z)
    C(z) = K_P + dt*K_I / (1 - z^-1)
    P(z) = (dt/T_sys) z^-1 / (1 - (1 - dt/T_sys) z^-1)   (T_sys = 0: z^-1)
    šum mezery dg -> g:                         S = 1 / (1 + L)
    šum proudu přepočtený na mezeru dI/(2*kappa*I_set) -> g:  T = L / (1 + L)
"""

import math

import numpy as np


def loop_gain(f, dt, kappa, K_P, K_I, T_sys):
    """Otevřený přenos L na frekvencích f [Hz] (komplexní pole)."""
    zi = np.exp(-2j * math.pi * np.asarray(f, dtype=float) * dt)   # z^-1
    C = K_P + dt * K_I / (1.0 - zi)
    if T_sys > 0:
        a = dt / T_sys
        P = a * zi / (1.0 - (1.0 - a) * zi)
    else:
        P = zi
    return 2.0 * kappa * C * P


def sensitivity(f, dt, kappa, K_P, K_I, T_sys):
    """Vrátí (S, T): přenos šumu mezery a šumu proudu (v mezeře) do g."""
    L = loop_gain(f, dt, kappa, K_P, K_I, T_sys)
    return 1.0 / (1.0 + L), L / (1.0 + L)


def closed_loop_poles(dt, kappa, K_P, K_I, T_sys):
    """Póly uzavřené smyčky (kořeny 1 + L(z) = 0) pro PI a T_sys > 0.

    (z - 1)(z - b) + 2*kappa*a*(K_P*(z - 1) + dt*K_I) = 0, a = dt/T_sys, b = 1 - a.
    """
    a = dt / T_sys
    b = 1.0 - a
    k = 2.0 * kappa * a
    return np.roots([1.0, -1.0 - b + k * K_P, b - k * K_P + k * dt * K_I])


def kp_stability_limit(dt, kappa, K_I, T_sys, K_hi=1.0):
    """Nejmenší K_P, při kterém má uzavřená smyčka pól |z| >= 1 (bisekce)."""
    lo, hi = 0.0, K_hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.max(np.abs(closed_loop_poles(dt, kappa, mid, K_I, T_sys))) >= 1.0:
            hi = mid
        else:
            lo = mid
    return hi


def psd_welch(x, dt, nseg):
    """Jednostranná PSD (Welch, Hann, překryv 50 %, odečtený průměr segmentu).

    Returns:
        (f, psd): frekvence [Hz] a PSD [jednotka^2/Hz].
    """
    x = np.asarray(x, dtype=float)
    win = np.hanning(nseg)
    norm = 2.0 * dt / (win ** 2).sum()
    acc = np.zeros(nseg // 2 + 1)
    count = 0
    for s in range(0, len(x) - nseg + 1, nseg // 2):
        seg = x[s:s + nseg]
        acc += np.abs(np.fft.rfft((seg - seg.mean()) * win)) ** 2
        count += 1
    if count == 0:
        raise ValueError("Záznam je kratší než jeden segment Welche")
    psd = acc * norm / count
    return np.fft.rfftfreq(nseg, dt), psd


def band_rms(f, psd, f_lo, f_hi):
    """rms v pásmu [f_lo, f_hi] z jednostranné PSD (součet přes biny)."""
    m = (f >= f_lo) & (f <= f_hi)
    return math.sqrt(psd[m].sum() * (f[1] - f[0]))
