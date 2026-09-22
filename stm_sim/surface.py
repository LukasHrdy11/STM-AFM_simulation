"""Modely povrchu: umělý schod (v1), schod s konečnou šířkou hrany (v2),
chaotický profil podle Mackey-Glassovy rovnice (MGE) a periodická mřížka
jednotlivých atomů (v3)."""

import numpy as np


def step_surface(x, x_edge, H):
    """Výška povrchu h(x) pro jeden nekonečně ostrý schod (v1).

    x < x_edge  -> 0
    x >= x_edge -> H
    """
    return H if x >= x_edge else 0.0


def ramp_surface(x, x_edge, H, w):
    """Výška povrchu h(x) pro schod s lineární rampou šířky w (v2).

    Přechod probíhá lineárně mezi x_edge - w/2 a x_edge + w/2:

    x <= x_edge - w/2                 -> 0
    x_edge - w/2 < x < x_edge + w/2   -> lineární nárůst 0 -> H
    x >= x_edge + w/2                 -> H

    Pro w -> 0 přechází v step_surface. Sklon rampy je s = H / w.

    Args:
        x: poloha hrotu podél skenu [m].
        x_edge: poloha středu hrany [m].
        H: výška schodu [m].
        w: šířka hrany [m], musí být > 0.

    Returns:
        Výška povrchu h [m].
    """
    x_start = x_edge - w / 2.0
    x_stop = x_edge + w / 2.0
    if x <= x_start:
        return 0.0
    if x >= x_stop:
        return H
    return H * (x - x_start) / w


def generate_mge_steps(n, x_start, spacing, H, min_frac=0.3):
    """Vygeneruje `n` hran `(x_edge, H_i)` pro `make_mackey_glass_surface`
    - uživatelské zadání "chci N schodů" bez ručního psaní seznamu.

    Hrany jsou rovnoměrně rozestavené od `x_start` s krokem `spacing`.
    Výšky `H_i` cyklicky variují mezi `min_frac*H` a `H` (modulace zlatým
    úhlem, deterministicky, bez RNG), aby povrch nebyl jen řada stejně
    vysokých schodů.

    Args:
        n: počet hran (>= 1).
        x_start: poloha první hrany [m].
        spacing: rozestup mezi sousedními hranami [m].
        H: maximální výška schodu [m].
        min_frac: nejmenší podíl H, na který výška schodu může klesnout.

    Returns:
        Seznam `n` dvojic `(x_edge, H_i)` vzestupně podle `x_edge`, přímo
        použitelný jako `steps` v `make_mackey_glass_surface`.
    """
    if n < 1:
        raise ValueError("n musí být >= 1")
    return [
        (x_start + i * spacing, H * (min_frac + (1 - min_frac) * ((i * 0.6180339887) % 1.0)))
        for i in range(n)
    ]


def _mackey_glass_series(n_out, beta=0.2, gamma=0.1, delay_steps=17,
                          power=10, x0=1.2, discard=1000):
    """Diskrétní (dt = 1) Eulerova integrace Mackey-Glassovy DDE:

        dx/dt = beta*x(t - tau) / (1 + x(t - tau)**power) - gamma*x(t)

    Se standardními parametry (beta=0.2, gamma=0.1, tau=17, power=10) je
    řešení chaotické. Historie pro t <= 0 je konstantní x0. Prvních
    `discard` kroků (přechodný jev z konstantní historie) se zahodí,
    zbytek se normalizuje na rozsah [0, 1].

    Returns:
        Seznam délky n_out s hodnotami v [0, 1].
    """
    total = n_out + discard
    xs = [x0] * total
    for i in range(1, total):
        i_delay = i - 1 - delay_steps
        x_delay = xs[i_delay] if i_delay >= 0 else x0
        xs[i] = xs[i - 1] + beta * x_delay / (1.0 + x_delay ** power) - gamma * xs[i - 1]
    serie = xs[discard:]
    lo, hi = min(serie), max(serie)
    rozsah = hi - lo if hi > lo else 1.0
    return [(v - lo) / rozsah for v in serie]


def make_mackey_glass_surface(steps, x_span_total, x_step, beta=0.2, gamma=0.1,
                               delay_steps=17, power=10, x0=1.2, discard=1000):
    """Vrátí funkci h(x) pro chaotický povrch podle Mackey-Glassovy rovnice (MGE),
    s libovolným počtem NEZÁVISLÝCH hran za sebou.

    `steps` je seznam dvojic `(x_edge, H)` seřazený vzestupně podle `x_edge`.
    Před `steps[0][0]` je povrch plochý (0, stejně jako step/ramp před hranou).
    Za hranou `i` (až do hrany `i + 1`, nebo do `x_span_total` za poslední
    hranou) sleduje normalizovanou (0..1) Mackey-Glassovu časovou řadu
    přeškálovanou na amplitudu `H_i`, vzorkovanou s krokem `x_step` [m]
    a lineárně interpolovanou mezi vzorky (viz _mackey_glass_series pro
    parametry rovnice). Každý úsek má vlastní, na ostatních nezávislou MG
    sekvenci (čerstvá historie s fázově posunutým `x0`) - úseky spolu
    díky citlivosti chaosu na počáteční podmínku nekorelují, přesto je
    celý povrch deterministický (žádné RNG).

    Args:
        steps: seznam `(x_edge, H)` [m, m], vzestupně podle `x_edge`.
        x_span_total: absolutní poloha konce posledního úseku [m]; musí být
            za poslední hranou a pokrýt celou očekávanou délku průjezdu
            (volající typicky dá x_span_total = |v| * t_end).
        x_step: vzorkovací krok profilu podél x [m], společný pro úseky.
        beta, gamma, delay_steps, power, x0, discard: parametry MG rovnice,
            viz _mackey_glass_series (x0 se pro každý úsek mírně posune).

    Returns:
        Funkce h(x) -> výška povrchu [m].
    """
    if not steps:
        raise ValueError("steps nesmí být prázdný seznam")
    edges = [x_edge for x_edge, _ in steps]
    if edges != sorted(edges):
        raise ValueError("steps musí být seřazené vzestupně podle x_edge")
    if x_span_total <= edges[-1]:
        raise ValueError("x_span_total musí ležet za poslední hranou v steps")

    konce_useku = edges[1:] + [x_span_total]
    useky = []  # (x_edge, x_grid, values)
    for i, ((x_edge, H), x_konec) in enumerate(zip(steps, konce_useku)):
        n_samples = int((x_konec - x_edge) / x_step) + 2
        serie = _mackey_glass_series(n_samples, beta=beta, gamma=gamma,
                                      delay_steps=delay_steps, power=power,
                                      x0=x0 + 0.037 * i, discard=discard)
        x_grid = [x_edge + k * x_step for k in range(n_samples)]
        values = [H * v for v in serie]
        useky.append((x_edge, x_grid, values))

    def h(x):
        if x <= useky[0][0]:
            return 0.0
        usek = useky[0]
        for kandidat in useky:
            if kandidat[0] <= x:
                usek = kandidat
            else:
                break
        _, x_grid, values = usek
        if x >= x_grid[-1]:
            return values[-1]
        return float(np.interp(x, x_grid, values))

    return h


def atoms_surface(x, x_start, H, spacing, sigma):
    """Výška povrchu h(x) pro periodickou mřížku Gaussových "atomů" (v3).

    Mřížka je NEKONEČNÁ (na rozdíl od MGE není co "doběhnout" na konci -
    volající, který chce sken konečné délky, si to řídí přes t_end/rozsah
    x, ne přes parametr téhle funkce). Středy vrcholů jsou v
    x_c(i) = x_start + (i + 0,5)*spacing pro i = 0, 1, 2, ... (odsazení o
    půl rozestupu, aby h(x_start) vyšlo blízko 0, ne na úbočí prvního
    atomu). Žádný explicitní clamp na 0 pro x < x_start - gaussovský ocas
    tam přirozeně vyjde ~0 a hard-clamp by navíc vytvořil nefyzikální
    skok přesně v x_start.

    h(x) = H * sum_i exp(-(x - x_c(i))^2 / (2*sigma^2))

    Suma se nepočítá přes celou (nekonečnou) mřížku, ale jen přes 2
    nejbližší sousedy z každé strany nejbližšího atomu (i_c ± 2, s
    ořezem i >= 0) - pro sigma <= spacing/6 (doporučené defaulty) je
    příspěvek NEJBLIŽŠÍHO souseda ~exp(-(spacing/sigma)^2/2) ~ 1,5e-8
    relativně k H (fyzikálně nevýznamné, ale ne numerická nula - proto
    má první atom mřížky, kterému ořez i >= 0 vezme levého souseda,
    hodnotu v maximu nepatrně jinou než vnitřní atomy o tenhle jeden
    chybějící příspěvek). Příspěvek druhého souseda (i_c ± 2) je řádu
    (1,5e-8)^~4, tedy skutečně numericky bezvýznamný.

    Args:
        x: poloha hrotu podél skenu [m].
        x_start: poloha, odkud mřížka atomů začíná [m] (první vrchol je
            v x_start + spacing/2).
        H: výška (amplituda) atomu [m].
        spacing: rozestup mezi sousedními atomy [m].
        sigma: šířka (směrodatná odchylka) Gaussova vrcholu [m]; má být
            výrazně menší než spacing (doporučeno spacing/6 nebo méně),
            jinak vrcholy vizuálně splynou v jednu vlnu.

    Returns:
        Výška povrchu h [m].
    """
    i_c = round((x - (x_start + 0.5 * spacing)) / spacing)
    h = 0.0
    for i in range(max(i_c - 2, 0), i_c + 3):
        x_ci = x_start + (i + 0.5) * spacing
        h += H * np.exp(-((x - x_ci) ** 2) / (2.0 * sigma ** 2))
    return h
