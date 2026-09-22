"""Korektnostní test: replikace obr. 5.11 (Voigtländer, Scanning Probe
Microscopy) - krokové odezvy P, I a PI regulátoru na skok setpointu.

Na rozdíl od STM smyčky (run_simulation.py) tu NENÍ žádná exponenciální
tunelová charakteristika - jde o obecné srovnání regulátorů na jednoduché
soustavě prvního řádu (FirstOrderPlant), přesně jak to dělá obr. 5.11
v literatuře. Chyba je tu přímo e = w - x (ne log-chyba), setpoint w
skočí v čase 0 z 0 na 1.

Testuje se, že implementace regulátorů v stm_sim/controller.py (stejné
třídy jako v STM simulaci) reprodukují kvalitativní podpisy z knihy:
- P: ustálí se pod setpointem (trvalá odchylka), na nulu se nedostane.
- I: setpoint dosáhne, ale znatelně pomaleji než PI.
- PI: setpoint dosáhne rychleji než samotné I.

Graf se ukládá do export/, žádné živé okno.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from stm_sim.controller import IntegralController, PIController, ProportionalController
from stm_sim.plant import FirstOrderPlant

EXPORT_DIR = Path(__file__).resolve().parent.parent / "export"

# --- Soustava a setpoint ---
T_sys = 1.0    # časová konstanta soustavy [s] (obecná jednotka, ne STM)
w = 1.0         # setpoint, skok 0 -> 1 v čase t = 0
dt = T_sys / 500
t_end = 30 * T_sys

# --- Zesílení regulátorů (voleno ilustrativně, kniha konkrétní hodnoty neudává) ---
# Zesílení I a PI jsou zvolená podtlumená (zeta < 1), aby odezva měla
# viditelný překmit (maximum, pak pokles k setpointu) - stejně jako v obr. 5.11.
# Zesílení jsou tu bezrozměrná (soustava i setpoint jsou v a.u.), rozumné
# rozsahy plynou ze vztahů níže (T_sys = 1, dt = T_sys/500):
#
# K_P  (samotné P): ustálí se na w*K_P/(1+K_P), tedy vždy pod setpointem.
#      Volit 0.5-10 (trvalá odchylka 67 % až 9 %). Nad ~100 už je odchylka
#      neviditelná a graf nic neukáže; nad 2*T_sys/dt = 1000 numericky
#      diverguje (Eulerova integrace).
# K_I  (samotné I): perioda i tlumení plynou z omega_n = sqrt(K_I/T_sys),
#      zeta = 1/(2*sqrt(K_I*T_sys)). Překmit (zeta < 1) vyžaduje
#      K_I > 1/(4*T_sys) = 0.25. Volit 0.3-20; nad ~100 je odezva prakticky
#      netlumená (překmit se blíží 100 %).
# PI:  zeta = (1+K_P_PI)/(2*sqrt(K_I_PI*T_sys)), takže P-složka tlumí.
#      Volit K_P_PI 0.2-5 a k tomu K_I_PI > (1+K_P_PI)^2/(4*T_sys), jinak
#      překmit zmizí. Aby test prošel, musí být PI rychlejší než samotné I,
#      tj. K_I_PI výrazně větší než K_I_SLOW.
K_P = 2.0        # pro samotný P
K_I_SLOW = 0.6    # pro samotné I - podtlumené, ale pomalejší než PI (menší K_I)
K_P_PI = 0.5       # PI: P-složka
K_I_PI = 1.2        # PI: I-složka - vyšší než u samotného I, proto rychlejší


def run(controller):
    plant = FirstOrderPlant(z0=0.0, T_sys=T_sys)
    x = 0.0
    n = int(t_end / dt)
    t = [0.0] * n
    xs = [0.0] * n
    for i in range(n):
        e = w - x
        y = controller.step(e, dt)
        x = plant.step(y, dt)
        t[i] = (i + 1) * dt
        xs[i] = x
    return t, xs


def time_to_reach(t, xs, fraction=0.9):
    """Čas, kdy x poprvé dosáhne fraction*w (None, pokud vůbec)."""
    threshold = fraction * w
    for ti, xi in zip(t, xs):
        if xi >= threshold:
            return ti
    return None


def main():
    t_p, x_p = run(ProportionalController(base=0.0, K_P=K_P))
    t_i, x_i = run(IntegralController(y0=0.0, K_I=K_I_SLOW))
    t_pi, x_pi = run(PIController(y0=0.0, K_I=K_I_PI, K_P=K_P_PI))

    final_p, final_i, final_pi = x_p[-1], x_i[-1], x_pi[-1]
    t90_i = time_to_reach(t_i, x_i)
    t90_pi = time_to_reach(t_pi, x_pi)

    checks = [
        ("P se neustálí na setpointu (trvalá odchylka)",
         abs(final_p - w) > 0.1),
        ("I nakonec dosáhne setpointu (do 2 %)",
         abs(final_i - w) < 0.02),
        ("PI dosáhne setpointu přesněji než I (do 1 %)",
         abs(final_pi - w) < 0.01),
        ("I dosáhne 90 % setpointu",
         t90_i is not None),
        ("PI dosáhne 90 % setpointu",
         t90_pi is not None),
        ("PI dosáhne 90 % setpointu rychleji než I",
         t90_i is not None and t90_pi is not None and t90_pi < t90_i),
    ]

    all_ok = True
    for description, ok in checks:
        print(f"[{'OK' if ok else 'SELHALO'}] {description}")
        all_ok &= ok

    print(f"\nP final={final_p:.3f}  I final={final_i:.3f} (t90={t90_i:.2f})  "
          f"PI final={final_pi:.3f} (t90={t90_pi:.2f})")
    print("VŠECHNY KONTROLY PROŠLY" if all_ok else "NĚKTERÁ KONTROLA SELHALA")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.axhline(w, color="black", linestyle="--", linewidth=1, label="Setpoint w")
    ax.plot(t_p, x_p, label="P-controller")
    ax.plot(t_i, x_i, label="I-controller")
    ax.plot(t_pi, x_pi, label="PI-controller")
    ax.set_xlabel("Time (a.u.)")
    ax.set_ylabel("Signal (a.u.)")
    ax.set_title("Replikace obr. 5.11 - kroková odezva regulátorů")
    ax.legend()
    fig.tight_layout()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPORT_DIR / "fig511.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Graf uložen do {out_path}")


if __name__ == "__main__":
    main()
