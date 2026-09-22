"""Regulátory (P, I, PI): počítají příkaz y z chyby e.

Regulátor sám o sobě neřídí fyzickou polohu hrotu - vrací příkaz y, který
dál zpracovává akční člen (viz plant.py, FirstOrderPlant). Bez tohoto
rozdělení by u P/PI regulátoru byl výstup použit přímo jako nová poloha
hrotu ve stejném kroku, ve kterém byl spočítán z chyby změřené na téže
poloze - P-složka by se pak ve skutečnosti chovala jako derivační člen,
ne proporcionální (protokol tohoto odvození je v README).

Regulátory jsou díky rozhraní step(e, dt) -> y nezávislé na fyzice STM -
používá je jak simulace STM (stm_sim/sim.py), tak obecný korektnostní
test step-response tří regulátorů (test_fig_5_11.py, replikace obr. 5.11
z Voigtländer, Scanning Probe Microscopy).
"""


def gain_from_tau(kappa, tau):
    """Zesílení K_I odpovídající zvolené časové konstantě I-smyčky STM.

    K_I = 1 / (2*kappa*tau)

    Args:
        kappa: rozpadová konstanta vlnové funkce [1/m].
        tau: požadovaná časová konstanta smyčky [s].

    Returns:
        Zesílení K_I [m/s] (rychlost hrotu na jednotku bezrozměrné chyby e).
    """
    return 1.0 / (2.0 * kappa * tau)


class ProportionalController:
    """y = base + K_P * e. Bez paměti - reaguje okamžitě, beze stavu."""

    def __init__(self, base, K_P):
        """
        Args:
            base: příkaz y při nulové chybě (e = 0).
            K_P: proporcionální zesílení (posun y na jednotku chyby e).
        """
        self.base = base
        self.K_P = K_P

    def step(self, e, dt):
        """Vrátí příkaz y odpovídající aktuální chybě e (dt se nepoužije)."""
        return self.base + self.K_P * e


class IntegralController:
    """dy/dt = K_I * e (diskrétně: y += dt * K_I * e)."""

    def __init__(self, y0, K_I):
        """
        Args:
            y0: počáteční příkaz y.
            K_I: integrační zesílení.
        """
        self.y = y0
        self.K_I = K_I

    def step(self, e, dt):
        """Provede jeden krok integrace a vrátí nový příkaz y."""
        self.y += dt * self.K_I * e
        return self.y


class PIController:
    """y = y_i + K_P * e, kde y_i je integrační stav: dy_i/dt = K_I * e.

    y_i (na rozdíl od výsledného y) neobsahuje P-korekci z předchozích
    kroků - proto se v dalším kroku nesčítá s už jednou aplikovanou
    P-korekcí. Právě tohle smíchání by P-člen změnilo na derivační
    (viz docstring modulu).
    """

    def __init__(self, y0, K_I, K_P):
        """
        Args:
            y0: počáteční příkaz y (při e = 0 splývá s počátečním y_i).
            K_I: integrační zesílení.
            K_P: proporcionální zesílení.
        """
        self.y_i = y0
        self.K_I = K_I
        self.K_P = K_P

    def step(self, e, dt):
        """Provede jeden krok PI regulace a vrátí nový příkaz y."""
        self.y_i += dt * self.K_I * e
        return self.y_i + self.K_P * e
