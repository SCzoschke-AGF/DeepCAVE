# Messung der 1.000-Trials-Grenze im Coverage-Planer

Gemessen wird ausschließlich die Strategie `minimize_trials` des bestehenden Planers. Es wird kein Sonderalgorithmus für maximale Zielstärke verwendet. Während dieser Messserie blieben die Produktionsdateien und die produktive Grenze unverändert. Die spätere Optimierung der Validierung ist in einer [separaten Vergleichsmessung](validation_optimization.md) dokumentiert; die folgenden Zahlen beziehen sich auf den ursprünglichen Stand.

In fünf der elf Fälle greift die Anzahlgrenze nicht; beide Varianten liefern jeweils denselben Plan. In fünf weiteren Fällen verhindert die Grenze vollständige Zielabdeckung, die ohne Grenze innerhalb des Zeitlimits erreicht wird. Im größten Fall sparen 1.000 statt rund 8.300 Vorschläge etwa 89 % CPU-Zeit, die Abdeckung bleibt dafür bei 29,4 % statt 95,3 %. Dort erreicht auch die unbeschränkte Variante das Ziel vor Ablauf des Suchzeitlimits nicht.

Die zusätzliche Laufzeit ohne Anzahlgrenze liegt beispielsweise bei 0,13 Sekunden für zwölf Parameter, acht Levels und Zielstärke 3; beim größeren Fall mit denselben Parametern und Zielstärke 4 sind es rund 36,7 Sekunden. Eine feste Anzahlgrenze ist deshalb kein einheitliches Rechenzeitbudget.

## Versuchsaufbau

- Hardware: Intel(R) Core(TM) i9-14900K; Python 3.10.19, NumPy 2.2.6, ConfigSpace 1.2.0.
- 11 Kombinationen aus synthetischem Suchraum und Zielstärke, jeweils 299 beobachtete erfolgreiche Trials. Jeweils zwei ganzzahlige Parameter, die übrigen kontinuierlich; pro Parameter gleich viele Bins. Wiederholte Konfigurationen bleiben in der Historie.
- 10 Fälle mit gleichverteilt gezogenen Bins; ein geclusterter Fall mit 70 % Wahrscheinlichkeit für den ersten Bin jedes Parameters. Beobachtungen sind zwischen den beiden Varianten identisch.
- Daten-Seed 20261021; fünf Planner-Seeds [0, 1, 2, 3, 4]. Je Seed wird die Reihenfolge der beiden Varianten abgewechselt.
- Ein nicht aufgezeichneter Warm-up-Lauf vor jeder Messserie. Imports und Datenaufbereitung sind nicht in den Laufzeiten enthalten.
- Die zehn Basisfälle und der zusätzliche Stresstest wurden in zwei aufeinanderfolgenden Messserien ausgeführt. Umgebungsdaten, Quellcode-Hashes, Daten-Seed, Planner-Seeds und Zeitlimit stimmen überein; die Rohdaten dokumentieren beide Serien.
- Identisches Suchzeitlimit von 30 Sekunden, identische Ressourcenlimits und identischer Algorithmus. Nur `MAX_ADDITIONAL_TRIALS` wird im Benchmark-Prozess zwischen 1.000 und unendlich umgeschaltet.
- Sequenzielle Ausführung; `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. Keine parallelen Benchmark-Läufe. Die CPU-Affinität wird nicht festgesetzt.
- Laufzeit: `perf_counter`; CPU-Zeit: `process_time`. Beides umfasst den gesamten Planeraufruf einschließlich Validierung, Materialisierung der Trial-Werte und interner Plan-ID-Erzeugung. Die abschließende Messung der JSON-Payload-Größe liegt außerhalb dieser Zeit.
- Zwei leichte Instrumentierungen zählen Suchdurchläufe und markieren den Beginn der Validierung. Eine Iteration entspricht einem Aufruf von `_candidates`; die Produktionslogik wird nicht ersetzt.
- Bei beidseitig vollständig erreichten Zielen wird für jeden gepaarten Seed zusätzlich geprüft, dass die `plan_id` identisch ist. Abweichende Zeiten bei solchen Fällen sind Messstreuung, keine Wirkung der Grenze.

## Ergebnisse

Alle Werte sind Mediane aus fünf Planner-Seeds. „6 × 4“ bedeutet sechs Parameter mit je vier Levels. Die CPU-Ersparnis ist `1 − Median(CPU mit Grenze) / Median(CPU ohne Grenze)`. Bei nicht greifender Grenze wird keine Prozentersparnis interpretiert.

| Suchraum | Zielstärke | Laufzeit mit 1.000er-Grenze | Laufzeit ohne Grenze | CPU-Ersparnis | Abdeckung mit → ohne | Vorschläge mit → ohne |
|---|---:|---:|---:|---:|---:|---:|
| 6 × 4 | 3 | 0.015 s | 0.015 s | Grenze greift nicht | 100.00 % → 100.00 % | 8 → 8 |
| 6 × 4 | 4 | 0.378 s | 0.379 s | Grenze greift nicht | 100.00 % → 100.00 % | 217 → 217 |
| 6 × 4 | 5 | 1.600 s | 1.794 s | 10.8 % | 98.21 % → 100.00 % | 1,000 → 1,096 |
| 6 × 4 | 6 | 1.377 s | 5.246 s | 73.8 % | 31.45 % → 100.00 % | 1,000 → 3,808 |
| 6 × 8 | 3 | 0.623 s | 0.595 s | Grenze greift nicht | 100.00 % → 100.00 % | 589 → 589 |
| 6 × 8 | 4 | 1.139 s | 6.994 s | 83.7 % | 31.47 % → 100.00 % | 1,000 → 6,078 |
| 8 × 4 | 4 | 0.377 s | 0.401 s | Grenze greift nicht | 100.00 % → 100.00 % | 348 → 348 |
| 12 × 8 | 3 | 1.703 s | 1.833 s | 7.1 % | 99.73 % → 100.00 % | 1,000 → 1,079 |
| 20 × 8 | 3 | 4.467 s | 6.297 s | 29.1 % | 97.73 % → 100.00 % | 1,000 → 1,486 |
| 8 × 4 (geclustert) | 4 | 0.536 s | 0.533 s | Grenze greift nicht | 100.00 % → 100.00 % | 478 → 478 |
| 12 × 8 | 4 | 4.424 s | 41.136 s | 89.2 % | 29.39 % → 95.31 % | 1,000 → 8,303 |

## Streuung und Abbruchgründe

| Suchraum / Zielstärke | Laufzeitbereich mit Grenze | Laufzeitbereich ohne Grenze | Abbruch mit Grenze | Abbruch ohne Grenze |
|---|---:|---:|---|---|
| 6 × 4 / t=3 | 0.014–0.030 s | 0.014–0.017 s | target_reached: 5 | target_reached: 5 |
| 6 × 4 / t=4 | 0.364–0.391 s | 0.362–0.414 s | target_reached: 5 | target_reached: 5 |
| 6 × 4 / t=5 | 1.540–1.649 s | 1.777–1.845 s | trial_limit: 5 | target_reached: 5 |
| 6 × 4 / t=6 | 1.323–1.388 s | 5.232–5.268 s | trial_limit: 5 | target_reached: 5 |
| 6 × 8 / t=3 | 0.604–0.634 s | 0.587–0.658 s | target_reached: 5 | target_reached: 5 |
| 6 × 8 / t=4 | 1.112–1.211 s | 6.872–7.053 s | trial_limit: 5 | target_reached: 5 |
| 8 × 4 / t=4 | 0.375–0.421 s | 0.389–0.421 s | target_reached: 5 | target_reached: 5 |
| 12 × 8 / t=3 | 1.658–1.739 s | 1.784–1.850 s | trial_limit: 5 | target_reached: 5 |
| 20 × 8 / t=3 | 4.329–4.497 s | 6.263–6.315 s | trial_limit: 5 | target_reached: 5 |
| 8 × 4 (geclustert) / t=4 | 0.519–0.545 s | 0.528–0.569 s | target_reached: 5 | target_reached: 5 |
| 12 × 8 / t=4 | 4.396–4.464 s | 41.059–41.488 s | trial_limit: 5 | timeout: 5 |

## Zeitanteile, CPU und Ergebnisgröße

„Suche“ umfasst die Suchschleife einschließlich anschließender Redundanzentfernung. „Validierung/Ausgabe“ beginnt mit dem Aufbau des unabhängigen Validierungszustands und endet bei Rückgabe des Plans. Das 30-Sekunden-Limit gilt im bestehenden Code für die Suchschleife; nachgelagerte Arbeit ist darin nicht eingeschlossen.

| Suchraum / Zielstärke | CPU mit → ohne | Suche mit → ohne | Validierung/Ausgabe mit → ohne | JSON-Payload mit → ohne |
|---|---:|---:|---:|---:|
| 6 × 4 / t=3 | 0.015 → 0.015 s | 0.010 → 0.010 s | 0.004 → 0.004 s | 0.01 → 0.01 MiB |
| 6 × 4 / t=4 | 0.378 → 0.379 s | 0.272 → 0.279 s | 0.104 → 0.102 s | 0.15 → 0.15 MiB |
| 6 × 4 / t=5 | 1.600 → 1.794 s | 1.131 → 1.273 s | 0.461 → 0.542 s | 0.69 → 0.73 MiB |
| 6 × 4 / t=6 | 1.376 → 5.246 s | 0.879 → 3.436 s | 0.480 → 1.799 s | 0.40 → 1.54 MiB |
| 6 × 8 / t=3 | 0.623 → 0.595 s | 0.262 → 0.244 s | 0.361 → 0.345 s | 0.50 → 0.50 MiB |
| 6 × 8 / t=4 | 1.139 → 6.994 s | 0.514 → 3.146 s | 0.638 → 3.847 s | 1.42 → 6.10 MiB |
| 8 × 4 / t=4 | 0.377 → 0.401 s | 0.187 → 0.197 s | 0.190 → 0.203 s | 0.43 → 0.43 MiB |
| 12 × 8 / t=3 | 1.703 → 1.833 s | 0.838 → 0.908 s | 0.845 → 0.894 s | 1.60 → 1.66 MiB |
| 20 × 8 / t=3 | 4.466 → 6.297 s | 3.194 → 4.426 s | 1.269 → 1.869 s | 1.98 → 2.83 MiB |
| 8 × 4 (geclustert) / t=4 | 0.536 → 0.533 s | 0.260 → 0.263 s | 0.265 → 0.269 s | 0.67 → 0.67 MiB |
| 12 × 8 / t=4 | 4.423 → 41.128 s | 3.093 → 30.004 s | 1.337 → 11.130 s | 2.12 → 17.44 MiB |

## Einordnung

- Eine generelle Ersparnis lässt sich aus einer festen Grenze nicht ableiten: Werden weniger als 1.000 Suchdurchläufe benötigt, liefern beide Varianten denselben Plan.
- Greift die Grenze, werden Zeit und Ergebnisvolumen reduziert. Dafür bleibt Zielabdeckung offen. Aus Suchdurchläufen allein lässt sich die tatsächliche Laufzeitersparnis nicht zuverlässig vorhersagen.
- Eine Anzahlgrenze begrenzt auch Validierung, Materialisierung und Ausgabe. Ein reines Suchzeitlimit ersetzt diese Schutzwirkung im aktuellen Code nicht vollständig.
- Vollständige Zielabdeckung bedeutet bei der heuristischen Suche nicht automatisch, dass die minimale Trial-Anzahl bewiesen wurde.
- Die absoluten Zeiten gelten für diese Maschine und diese synthetischen Daten. Diese Fallauswahl ist kein repräsentativer Querschnitt realer Studien und begründet keinen universellen Durchschnittswert. Queue, Browserdarstellung und Ausführung der vorgeschlagenen Optimierungs-Trials sind nicht gemessen.

## Reproduktion

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/codex-uv-cache \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --locked python examples/api/benchmark_coverage_planning.py \
  --output docs/validation/coverage_planning/results.json
```

Der erste Warm-up-Lauf wird nicht gespeichert. `--cases`, `--seeds` und `--timeout` erlauben gezielte Wiederholungen. Für abweichende Parameter einen anderen Ausgabepfad verwenden.

- [Rohmessungen und Umgebung](results.json)
- [Aggregierte Messwerte](summary.json)
- [Benchmark-Skript](../../../examples/api/benchmark_coverage_planning.py)
