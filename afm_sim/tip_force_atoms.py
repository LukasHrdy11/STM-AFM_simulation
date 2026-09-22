"""Povrch "atoms-AFM": periodická řada bodových atomů (jen pro AFM).

Na rozdíl od `stm_sim.surface.atoms_surface` (spojitá výška h(x) ze součtu
Gaussových kopečků) je povrch tady JEN množina bodů. Nemá výšku, kterou by hrot
sledoval - Δf vzniká sečtením sil od jednotlivých atomů ve skutečných 3D
vzdálenostech. Proto AFM "vidí" atom i tehdy, když hrot není přímo nad ním:
boční dosah není parametr, vychází z dosahu párové síly.

Síla mezi hrotem (bodový apex) v poloze (x, y_scan, z) a atomem i v poloze
(x_i, 0, z_a) je LJ síla podél spojnice, `lj_force(r_i)`, kde
`r_i = sqrt((x - x_i)^2 + y^2 + (z - z_a)^2)`. Kmitání cantileveru je ale svislé,
takže do Δf (rov. 17.15) vstupuje jen SVISLÁ složka:

    F_z(x, z) = sum_i lj_force(r_i) * (z - z_a) / r_i

Faktor `(z - z_a)/r_i` je průmět na osu z. S jedním atomem přímo pod hrotem
(x = x_i, y = 0) je `(z - z_a)/r_i = 1` a F_z se redukuje na `lj_force(z - z_a)`,
tedy na dosavadní model (viz tests/test_atoms_afm.py).

Konvence znaménka jako u `lj_force`: kladné F = odpudivé (tlačí hrot nahoru).

Sečtou se VŠECHNY atomy řady (N je malé, řádově desítky), takže není žádný
odříznutý cutoff. Pozor při rozšíření o dalekodosahové členy (vdW koule,
elektrostatika): ty jsou geometrie celého hrotu, ne párové síly atomů, a do tohoto
součtu se nedají jednoduše přičíst.
"""

import numpy as np

from afm_sim.tip_force import lj_force


class AtomRow:
    """Řada `n` bodových atomů v rovině z = `z_atom`, na přímce y = 0.

    Středy atomů: x_i = x_start + (i + 0,5)*spacing (odsazení o půl rozestupu, stejně
    jako u `atoms_surface`, aby sken začínal mezi atomy).

    Args:
        x_start: poloha, odkud řada začíná [m].
        spacing: rozestup atomů [m].
        n: počet atomů.
        z_atom: výška roviny atomů [m] (referenční rovina, vzdálenost d = z_tip - z_atom).
        y_offset: boční posun skenovací linky od řady atomů [m] (0 = hrot jede přímo nad
            atomy; > 0 = linka vedle řady, hrot nad atom nikdy nepřijde).
    """

    def __init__(self, x_start, spacing, n, z_atom=0.0, y_offset=0.0):
        if n < 1:
            raise ValueError("n musí být >= 1")
        if spacing <= 0:
            raise ValueError("spacing musí být > 0")
        self.x = x_start + (np.arange(n) + 0.5) * spacing
        self.spacing = spacing
        self.n = n
        self.z_atom = z_atom
        self.y_offset = y_offset

    def distances(self, x, z):
        """Vzdálenosti hrotu v (x, y_offset, z) od všech atomů; tvar (len(z), n)."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        dx = x - self.x[None, :]
        dz = (z - self.z_atom)[:, None]
        return np.sqrt(dx ** 2 + self.y_offset ** 2 + dz ** 2), dz

    def force_z(self, x, z, U0, Ra):
        """Svislá složka celkové síly na hrot v poloze (x, z) [N] (kladná = odpudivá).

        `z` může být pole (rov. 17.15 volá sílu v celém oscilačním cyklu); výstup má
        stejný tvar jako `z`.
        """
        z_arr = np.asarray(z, dtype=float)
        r, dz = self.distances(x, z_arr)
        f = np.sum(lj_force(r, U0, Ra) * dz / r, axis=1)
        return f if z_arr.ndim else float(f[0])

    def nearest_distance(self, x, z):
        """Vzdálenost hrotu (střední poloha) od nejbližšího atomu [m]."""
        r, _ = self.distances(x, z)
        return float(np.min(r))
