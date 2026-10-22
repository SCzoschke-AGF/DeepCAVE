# Vergleich der Validierung vor und nach der Optimierung

Die Gültigkeit konkreter Parameterwerte, ihre Schrittweiten und ihre Rückübersetzung in die geplanten Stufen werden weiterhin geprüft. Die Materialisierung liefert jetzt zusätzlich die bereits geprüften Stufen zurück. Der aufrufende Planer wiederholt Gültigkeitsprüfung und Rückübersetzung nicht mehr. Die Anzahl fehlender Kombinationen wird anhand des unabhängig neu ermittelten Coverage-Gewinns fortgeschrieben, statt für jeden Trial das gesamte Missing-Array zu summieren.

## Messmethode

- Derselbe Stresstest wie im ursprünglichen Benchmark: zwölf Parameter, acht Levels, Zielstärke vier; 299 synthetische Beobachtungen, Daten-Seed 20261021 und Planner-Seed 0.
- Eine echte Suche der ursprünglichen Version erzeugt 8.303 Vorschläge. Für diese Erfassung wird ausschließlich im Benchmark die Anzahlgrenze auf 8.303 und das Suchzeitlimit auf 120 Sekunden gesetzt. Die Produktionsgrenze bleibt bei 1.000.
- Anschließend erhalten beide Implementierungen exakt diese Vorschläge, einmal die ersten 1.000 und einmal alle 8.303. Kandidatensuche und lokale Verbesserung werden für diese Wiederholungen durch die aufgezeichnete Folge ersetzt. Gemessen wird ausschließlich die anschließende Phase vom Aufbau des unabhängigen Validierungszustands bis zur Rückgabe des Plans. Die ursprüngliche Suche und die wiederholte Übernahme der Vorschläge sind nicht Teil dieser Zeit.
- Drei Wiederholungen je Umfang und Implementierung; Reihenfolge vorher/nachher abwechselnd. Sequenzielle Ausführung mit denselben Thread-Einstellungen wie im ursprünglichen Benchmark. Es wird kein neuer allgemeiner Durchschnitt über Suchräume behauptet.
- Alle zugehörigen Plan-IDs stimmen zwischen vorher/nachher und den Wiederholungen überein. Für 8.303 Vorschläge stimmt die ID auch mit dem ursprünglichen Suchlauf überein. Die IDs umfassen die konkreten Trial-Werte und den gesamten Coverage-Verlauf.
- Die ursprünglichen Module werden aus Git-Revision `456feb158ffda058505ee43a57db79ddd9cf6cbf` geladen. Maschineninformationen und Quellcode-Hashes beider Versionen stehen in den Rohdaten.

## Ergebnis

Mediane der Zeit für Validierung, Materialisierung und Ergebniserzeugung, einschließlich interner Plan-ID-Erzeugung. Imports, Datenaufbereitung und die externe Messung der JSON-Größe sind ausgeschlossen.

| Vorschläge | Vorher | Nachher | Zeitersparnis | Bereich vorher | Bereich nachher |
|---:|---:|---:|---:|---:|---:|
| 1.000 | 1,368 s | 0,492 s | 64,0 % | 1,352–1,373 s | 0,491–0,504 s |
| 8.303 | 11,028 s | 4,201 s | 61,9 % | 10,960–11,375 s | 4,186–4,271 s |

Die Nachverarbeitung des großen Plans benötigt damit rund 6,8 Sekunden weniger. Die Suche selbst wird durch diese Änderung nicht beschleunigt. Das Suchzeitlimit begrenzt weiterhin nur die Suche; die verbleibende Nachverarbeitung kommt anschließend hinzu.

## Reproduktion

```bash
PYTHONPATH=. UV_CACHE_DIR=/tmp/codex-uv-cache \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --locked python examples/api/benchmark_coverage_validation.py \
  --baseline-ref 456feb158ffda058505ee43a57db79ddd9cf6cbf \
  --output /tmp/coverage_validation_comparison.json
```

- [Rohmessungen](validation_optimization.json)
- [Vergleichsskript](../../../examples/api/benchmark_coverage_validation.py)

Die 33 Tests in `tests/test_evaluators/test_coverage_planning.py` bestehen. Sie prüfen unter anderem Coverage-Verläufe anhand unabhängig aufgezählter Kombinationen, gemischte Parametertypen, leere und vollständige Pläne sowie die Abweisung falscher Schrittweiten und abweichender Rückübersetzungen.
