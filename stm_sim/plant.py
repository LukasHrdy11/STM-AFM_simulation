"""Akční člen (piezo) mezi příkazem regulátoru y a skutečnou polohou hrotu z.

T_sys * dz/dt = y - z

Regulátor vrací příkaz y (viz controller.py); Plant určuje, jak rychle se
tomuto příkazu skutečně přizpůsobí fyzická poloha hrotu z_tip. Bez tohoto
mezičlánku by u regulátoru s P-složkou byl výstup y použit přímo jako nová
poloha hrotu ve stejném kroku, ve kterém byl spočítán z chyby změřené na
téže poloze - to vede k patologickému chování (P-složka by se ve
skutečnosti chovala jako derivační člen, ne proporcionální, viz diskuze
v README).
"""


class FirstOrderPlant:
    """Prvního řádu zpožděný akční člen: T_sys*dz/dt = y - z.

    T_sys = 0: z sleduje y okamžitě (zjednodušení v1 - "bez setrvačnosti
    pieza"). Postačuje pro čistě integrační regulátor (I), kde stačí,
    aby P-složka regulátoru chyběla.

    T_sys > 0: konečná rychlost odezvy akčního členu. Nutné, má-li mít
    P-složka regulátoru na chování smyčky jiný vliv než jen přeškálování
    efektivního zesílení (viz README, "Poznatek ze skenu H/v/tau").
    """

    def __init__(self, z0, T_sys=0.0):
        """
        Args:
            z0: počáteční poloha z [m] (nebo jiná fyzikální veličina
                v obecném použití, viz test_fig_5_11.py).
            T_sys: časová konstanta akčního členu [s]. 0 = okamžitá odezva.
        """
        self.z = z0
        self.T_sys = T_sys

    def step(self, y, dt):
        """Provede jeden krok akčního členu a vrátí novou polohu z.

        Args:
            y: příkaz regulátoru (žádaná poloha).
            dt: délka časového kroku [s].

        Returns:
            Nová poloha z.
        """
        if self.T_sys <= 0:
            self.z = y
        else:
            self.z += dt * (y - self.z) / self.T_sys
        return self.z
