
- Analyse von Fluktuationssensitivität
- Configuratoin analyse: Un-/Abhängigkeit einzelner Parameter von anderen Parametern (zB wenn ich sehe PAR1 hat einen Sweet Spot, ist das vllt Konfundiert weil es immer nur zusammen mit einem bestimmten performance-relevanten Wert bei einem anderen Parameter zusammen aufgetaucht ist)

- Einen BEreich um sich die Trials als Tabelle anzusehen und zu filtern nach Metriken

- Ab welcher Trialzahl kann man die KDE auch auf Top 10 vs Top 90 laufen lassen?


1. Realisierte Modelleigenschaften speichern
Künftig erfassen wir beim Training für jeden CV-Fold zusätzlich:
- Tatsächliche Baumtiefe (Maximum, Mittelwert)
- Anteil der Bäume am konfigurierten Tiefenlimit
- Tatsächliche Anzahl der Bäume
- Anzahl der Blätter und Splits
- Anzahl tatsächlich verwendeter Features
- Optional: Modellgröße und weitere Komplexitätsmaße
Speicherung: Als numerische Optuna-user_attrs, aggregiert über die fünf Folds. Die Einzelwerte pro Fold können wir zusätzlich kompakt speichern.

Analyse: Über die bereits vorhandenen Multimetrik- und 2D-Analysen in Lighthouse. Keine neue Analyseoberfläche erforderlich.

- über coverage erhobene  trials per checkbox aus den analysen ausschließen