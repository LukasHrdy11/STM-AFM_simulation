# Simulace zpětné vazby STM a FM-AFM — interaktivní demonstrace

Výukový nástroj k rastrovací sondové mikroskopii: simuluje zpětnovazební
smyčku tunelového mikroskopu (STM, režim konstantního proudu) a
frekvenčně modulovaného AFM (qPlus) a nechává si s ní hrát — posuvníkem,
ne editací kódu.

## Spuštění

```bash
git clone <adresa tohoto repa>
cd <složka repa>
pip install --user -r requirements.txt
jupyter notebook interactive/dashboard.ipynb
```

Pak Cell → Run All a dál už jen posouvat ovladače. Nic se nikam neposílá,
všechno běží lokálně.

## Co je uvnitř

| Složka | Obsah |
|---|---|
| `stm_sim/` | fyzika STM smyčky: regulátory, akční člen, povrchy, tunelový proud, šum |
| `afm_sim/` | fyzika FM-AFM: posun frekvence, silové modely hrotu (LJ, vdW koule, elektrostatika), šum |
| `interactive/` | interaktivní panely a `dashboard.ipynb` (jediný vstupní bod) |
| `tests/` | korektnostní testy; spouští se přímo, vypisují `[OK]`/`[SELHALO]` |

Dávkové spouštěče `run_simulation.py` a `run_simulation_afm.py` počítají
totéž a ukládají statické PNG — hodí se, když je potřeba obrázek do
prezentace. Že panel a spouštěč počítají bit-přesně totéž, hlídá
`tests/test_interactive.py`.

## Ověření, že to funguje

```bash
python3 tests/test_interactive.py   # interaktivní vrstva vs. dávkový spouštěč
python3 tests/test_fig_5_11.py      # regulátory P/I/PI proti literatuře
python3 tests/test_open_loop.py     # režim bez zpětné vazby + krokování
python3 tests/test_sum.py           # šum v simulaci
```

## Meze modelu (důležité pro závěry)

- **Reálná je jen dynamika smyčky, ne absolutní vzdálenosti.** Chyba
  `e = ln(I/I_set) = -2·kappa·(g - g_set)` konstantu předexponenciálu
  vykrátí, takže `g_set` i `g_contact` jsou odhady z literatury. Závěry
  typu „bezpečná vzdálenost je X nm" model neunese.
- **U STM je změřená `kappa`** (z naměřené závislosti proudu na Z) **a
  `K_P`** (z hlaviček přístroje), plus velikost šumu. Ostatní hodnoty
  jsou nastavení měření.
- **U AFM není kalibrováno nic kromě šumu.** Pokus vyfitovat parametry
  hrotu (U0, Ra) proti naměřené Δf(z) skončil negativně — atomární
  Lennard-Jonesův model neumí vysvětlit, jak pomalu naměřená Δf klesá.
  Hodnoty v `afm_sim/measured.py` jsou ilustrativní odhady a panel to
  u sebe píše.
- **Některé meze jsou numerické, ne fyzikální.** Mez stability `K_P` je
  artefakt Eulerova kroku, ne vlastnost přístroje.

Původ každé konstanty je v komentáři u ní: `stm_sim/measured.py`,
`afm_sim/measured.py`, `stm_sim/noise.py`, `afm_sim/noise.py`.

## Literatura

Výklad k panelům: B. Voigtländer, *Scanning Probe Microscopy* (Springer).
Kapitola 5.7–5.9 (zpětnovazební regulátor a jeho implementace v STM),
kapitola 17.1.1 a 17.2.2 (FM mód, vzorec pro Δf, sledování PLL),
kapitola 18 (šum v AFM), kapitola 20.6 (režim konstantního proudu vs.
konstantní výšky).

## Licence

MIT (viz `LICENSE`) — použití ve výuce i úpravy jsou vítané.

Autor: Lukáš Hrdý. Nástroj vznikl jako část práce o zpětnovazební smyčce
STM; naměřená data a analýzy k ní zůstávají v samostatném (neveřejném)
repozitáři, tady jsou jen výsledné konstanty s uvedením původu.
