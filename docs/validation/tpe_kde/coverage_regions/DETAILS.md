# Lighthouse: Scoring in fehlenden und schwach besetzten Coverage-Regionen

Ausgeführt am **23. Oktober 2026**. **Phase A ist vollständig abgeschlossen.**
Die optimierte KDE behält ihren mittleren Rangkorrelationsvorsprung auch bei
Testkonfigurationen, die bisher fehlende Kombinationen schließen. Bei XGBoost
sinkt ihre Korrelation gegenüber Regular etwas; bei CatBoost bleibt sie erhalten.
**Phase B wurde vorbereitet, aber wegen fehlender GPU-Ressourcen nicht trainiert.**
Eine prospektive Bestätigung liegt deshalb noch nicht vor.

## Primärer Vergleich

Mittelwerte der fünf Fold-Korrelationen, keine Korrelation gepoolter absoluter
Scores. N zählt unterschiedliche tatsächlich bewertete Konfigurationen über alle
Folds. Beide Verfahren bewerten innerhalb jeder Zeile dieselben Konfigurationen.

| Studie | Testpopulation | KDE Spearman ρ | TPE Spearman ρ | N evaluiert |
|---|---|---:|---:|---:|
| XGBoost | Regular, retrospektiv | 0,857 | 0,630 | 1.350 |
| XGBoost | Coverage-seeking, retrospektiv | 0,823 | 0,603 | 650 |
| XGBoost | Coverage-seeking, prospektiv | nicht ausgeführt | nicht ausgeführt | 0; 25 vorbereitet |
| CatBoost | Regular, retrospektiv | 0,919 | 0,916 | 896 |
| CatBoost | Coverage-seeking, retrospektiv | 0,924 | 0,906 | 1.104 |
| CatBoost | Coverage-seeking, prospektiv | nicht ausgeführt | nicht ausgeführt | 0; 25 vorbereitet |

![Rankingqualität nach Coverage-Situation](ranking_by_coverage.png)

Die Punkte zeigen einzelne Folds, die Rauten Mittelwerte und die Balken die
deskriptive Fold-SD. Die grauen Linien verbinden die gepaarten Methodenwerte.
[Vektorgrafik](ranking_by_coverage.svg) · [Vergleichstabelle als CSV](comparison.csv)

## Versuchsaufbau und Definition der Testpopulationen

Verwendet wurden dieselben Random-Studien mit je 2.000 Trials: XGBoost,
Studien-ID 1, zehn Parameter; CatBoost, Studien-ID 6, vier Parameter. Datenquelle
ist `../streamingmodel_NonRouterHHs/optuna.db`, über den bisherigen Loader
schreibgeschützt geöffnet und in eine temporäre Datenbank kopiert.
Die bisherigen fünf Folds mit Seed 42 bleiben unverändert: je 1.600 historische
und 400 zurückgehaltene Trials. Keine neue HPO, keine UI- oder Methodenänderung.

KDE-Kernel, Transformationen, beschränkte LOO-Bandbreitenoptimierung, Startwerte,
Grenzen und Abbruchregeln bleiben unverändert. TPE verwendet den bestehenden
multivariaten Adapter, uniforme Gewichte und die bisherige Prior-Konfiguration.
Beide erhalten dieselben chronologisch geordneten Good-/Bad-Gruppen mit 20/80;
Optunas Standard-Gamma wird dabei umgangen. Die logarithmierten Dichtequotienten
aller 4.000 retrospektiven Testkonfigurationen reproduzieren die bisherigen
TPE- und KDE-Scores **exakt**, maximale Abweichung jeweils null.

Die Coverage-Analyse verwendet unverändert Lighthouse `CoverageSpec`, dessen
Deklarationsgrenzen, Log-Transformationen und diskrete Schrittweiten sowie
`CoverageState`. Untersucht werden sechs Sichten: Auflösungen 4 und 8, jeweils
mit Strength 2, 3 und 4. Endliche Parameter können weniger Stufen besitzen;
bei CatBoost hat `depth` auch bei Auflösung 8 nur fünf Stufen.

Je Fold entstehen die Belegungszahlen ausschließlich aus seinen Trainingstrials:

- **Coverage-seeking:** mindestens eine aktuell fehlende t-Wege-Kombination.
  Der Gain zählt Kombinationen, die dieser einzelne Kandidat gegenüber dem
  unveränderten historischen Stand zusätzlich schließen würde.
- **Regular:** alle projizierten Kombinationen sind bereits vorhanden.
  Das bedeutet ausdrücklich nicht, dass jede davon häufig beobachtet wurde.
- **Selten, aber vorhanden:** Belegung 1–5. Diese Zusatzdiagnostik zählt auf
  demselben Coverage-Gitter, ohne Coverage- oder Scoring-Verfahren zu verändern.
  Unter Kandidaten ohne fehlende Kombinationen werden hohe und niedrige
  Seltenheitsexposition anhand oberer/unterer Quartile einer Referenz getrennt.
  Bei gleichen Quartilgrenzen wird die obere Gruppe strikt abgegrenzt; es wird
  keine künstliche Aufteilung identischer Coverage-Werte erzwungen.

Für die primäre Sicht werden vor Berechnung der Qualitätsmetriken je Fold
5.000 theoretisch zufällige Referenzkonfigurationen mit Seed 20261023 verwendet.
Bevorzugt werden Sichten mit 15–85 % Coverage-seeking-Referenzpunkten in jedem
Fold, anschließend mit möglichst ausgeglichenen Gruppengrößen. Objectives und
KDE-/TPE-Scores gehen weder in diese Wahl noch in die Gruppenzuordnung ein.
Alle sechs Sichten bleiben für die Sensitivitätsanalyse gespeichert.

Das wählt **Auflösung 4 / Strength 4 für XGBoost** und **Auflösung 8 / Strength 4
für CatBoost**. XGBoost besitzt dabei 210 Parametergruppen und 53.760 mögliche
projizierte Kombinationen; trotz 99,72–99,77 % historischer Tupel-Coverage
schließen 650 Testpunkte mindestens eine Lücke. CatBoost besitzt eine vollständige
Vier-Parameter-Projektion mit 2.560 Zellen; deren Coverage beträgt 45,27–46,99 %.
Die beiden primären Sichten untersuchen daher unterschiedliche Coverage-Geometrien.

## Gepaarte Unterschiede und Monotonie

Alle Differenzen werden innerhalb desselben Folds berechnet. Δρ ist TPE minus
KDE; negative Werte sprechen für KDE. SD beschreibt die fünf abhängigen Folds.

| Studie | Population | Δρ, Mittel ± SD | Beobachtete Spannweite |
|---|---|---:|---:|
| XGBoost | Regular | −0,228 ± 0,037 | −0,276 bis −0,183 |
| XGBoost | Coverage-seeking | −0,219 ± 0,026 | −0,259 bis −0,194 |
| CatBoost | Regular | −0,003 ± 0,006 | −0,013 bis +0,002 |
| CatBoost | Coverage-seeking | −0,018 ± 0,003 | −0,020 bis −0,013 |

KDE hat in beiden Coverage-seeking-Populationen in jedem der fünf Folds das
höhere ρ. Bei CatBoost/Regular liegen die Verfahren nahe beieinander.

Die Scorekurven verwenden pro Studie, Methode, Population und Fold dieselbe
Anzahl Gruppen: **fünf bei XGBoost, sieben bei CatBoost**, jeweils mindestens
20 Beobachtungen pro Gruppe. Damit wird die kleinere Population nicht durch
eine gröbere Darstellung bevorzugt. Es sind hier keine Dezilkurven. Unter drei
tragfähigen Gruppen wird ganz auf eine Kurve verzichtet. Die Gruppen entstehen
aus fold- und populationslokalen Score-Rängen mit stabiler Trialnummer bei Ties.

| Studie / Population | Monotone Fold-Kurven KDE / TPE | Mittlere AP-Spreizung KDE / TPE |
|---|---:|---:|
| XGBoost / Regular | 5/5 / 4/5 | 0,02628 / 0,01782 |
| XGBoost / Coverage-seeking | 5/5 / 5/5 | 0,02531 / 0,01687 |
| CatBoost / Regular | 5/5 / 5/5 | 0,02035 / 0,01900 |
| CatBoost / Coverage-seeking | 5/5 / 5/5 | 0,02078 / 0,01782 |

Spreizung ist die mittlere tatsächliche AP der höchsten minus niedrigsten
Scoregruppe. Die einzige Abnahme in diesen Einzelkurven betrifft XGBoost/TPE
Regular, um 0,000124 AP. Auf dieser gröberen Auflösung allein trennt Monotonie
die Verfahren wenig; die Rangkorrelation und die Rohbeobachtungen zeigen weiterhin
Unterschiede. Monotonie von fünf Gruppen garantiert keine Monotonie jeder feineren
Auflösung und auch keine fehlerfreie Bewertung einzelner Konfigurationen.

![XGBoost: Beobachtungen und Score-Performance-Kurven](xgboost/score_performance.png)

![CatBoost: Beobachtungen und Score-Performance-Kurven](catboost/score_performance.png)

Dünne Linien: einzelne Folds; dicke Linien: Mittel ihrer Gruppenmittelwerte.
Die Punktwolken zeigen die individuellen AP-Werte über lokalen Score-Rängen.
Absolute Scores verschiedener Folds werden nicht zusammengeführt.

## Seltene Kombinationen und Sensitivität

Die zusätzliche Analyse bei Auflösung 4 / Strength 4 trennt ausschließlich
bereits belegte Kombinationen nach Seltenheitsexposition; beide Gruppen
schließen **keine** vollständig fehlenden Kombinationen. „Gering“ und „hoch“
beziehen sich auf die beschriebenen Referenzquartile, nicht auf vollständig
dicht beziehungsweise vollständig dünn besetzte Konfigurationen.

| Studie | Seltenheitsexposition | KDE ρ | TPE ρ | N |
|---|---|---:|---:|---:|
| XGBoost | gering | 0,887 | 0,670 | 332 |
| XGBoost | hoch | 0,819 | 0,587 | 368 |
| CatBoost | gering | 0,928 | 0,918 | 1.201 |
| CatBoost | hoch | 0,913 | 0,893 | 789 |

Beide Verfahren verlieren hier etwas Rangkorrelation, KDE bleibt vorn. Für
XGBoost/geringe Exposition reicht ein Fold nicht für drei Gruppen mit mindestens
20 Beobachtungen; dort werden nur Korrelationen, keine künstlich präzisen Kurven
berichtet. In der primären CatBoost-Sicht lässt die eine Projektion keine
sinnvolle Quartiltrennung allein nach der Zahl seltener Projektionen zu;
deshalb wird die ergänzende gröbere Sicht separat ausgewiesen.

Bei Strength 2 sowie Auflösung 4 / Strength 3 gibt es in beiden Studien keine
fehlenden Kombinationen unter den Testpunkten. Für feinere Sichten ergibt sich:

| Studie | Auflösung / Strength | N Coverage-seeking | KDE ρ | TPE ρ | Einschränkung |
|---|---|---:|---:|---:|---|
| XGBoost | 8 / 3 | 1.949 | 0,847 | 0,620 | Regular nur 51 Punkte; kein tragfähiger Gegenvergleich |
| XGBoost | 8 / 4 | 2.000 | 0,848 | 0,622 | alle Punkte Coverage-seeking |
| CatBoost | 4 / 4 | 10 | nicht berichtet | nicht berichtet | je Fold weniger als zehn Lückenschließer |
| CatBoost | 8 / 3 | 173 | 0,891 | 0,885 | kleine Fold-Gruppen, nur explorative Zusatzsicht |

Somit ist die Coverage-Definition für die Interpretierbarkeit entscheidend.
Ein gesättigtes Gitter kann Lücken nicht untersuchen; ein zu feines Gitter kann
alle Konfigurationen als lückenschließend einstufen. Kleine Zusatzgruppen
begründen keine allgemeine Methodenüberlegenheit.

## Zusammenhang mit Scoringfehler und Einordnung

Als skalenunabhängige Fehlerdiagnostik dient der absolute Abstand zwischen
Score-Rang und AP-Rang, jeweils innerhalb aller 400 Testpunkte eines Folds auf
[0,1] skaliert. Das ist ein Rankingfehler, keine AP-Prognoseabweichung.

| Studie | Verfahren | Mittleres ρ(Gain, Rangfehler) | Rangfehler Regular / Coverage-seeking |
|---|---|---:|---:|
| XGBoost | KDE | +0,030 | 0,116 / 0,125 |
| XGBoost | TPE | +0,005 | 0,197 / 0,197 |
| CatBoost | KDE | +0,004 | 0,084 / 0,084 |
| CatBoost | TPE | +0,064 | 0,090 / 0,096 |

Die Zusammenhänge sind klein. Bei XGBoost/KDE ist das Vorzeichen nicht über
alle Folds konsistent. Bei CatBoost ist Gain in der primären Sicht nur 0 oder 1;
hier wird keine kontinuierliche Dosis-Wirkungs-Beziehung untersucht.
Die vollständigen Ergebnisse für Gain und seltene Kombinationen stehen in
`coverage_rank_errors.csv` je Studie.

Die zentralen Fragen lassen sich damit wie folgt beantworten:

1. **Hohe KDE-Korrelation erhalten?** Ja, retrospektiv 0,823 beziehungsweise 0,924
   in den primären Coverage-seeking-Populationen.
2. **Mehr oder weniger Verlust als TPE?** Bei XGBoost sinkt KDE im Mittel um
   0,035 und TPE um 0,026 gegenüber Regular; KDE verliert dort numerisch etwas
   mehr, bleibt aber klar vorn. Bei CatBoost steigt KDE um 0,005, während TPE
   um 0,010 sinkt. Diese Unterschiede sind deskriptiv und foldabhängig.
3. **Versagen beide in bestimmten Regionen?** Ein belastbarer Bereich mit
   systematischem Versagen beider Verfahren wurde nicht identifiziert.
   Seltene Kombinationen sind anspruchsvoller; für extreme Lücken und kleine
   Gruppen bleibt die Evidenz begrenzt.
4. **Gain als Fehlerindikator?** Kein starker, konsistenter Zusammenhang.
   Ein größerer Coverage-Gewinn ist hier kein verlässlicher Ersatz für eine
   Unsicherheitsangabe zum Score.
5. **Prospektive Übereinstimmung?** Noch offen; neue Konfigurationen wurden
   bewertet im Sinne von Scores, aber nicht mit dem tatsächlichen Objective trainiert.
6. **Lighthouse-Einsatz?** Die KDE ist ein gut gestützter Kandidat zur zusätzlichen
   Performance-Bewertung von Coverage-Vorschlägen innerhalb dieser Suchräume.
   Eine uneingeschränkte Freigabe für praktisch unbeobachtete Regionen folgt
   daraus noch nicht. Der kontrollierte prospektive Test bleibt dafür notwendig.

Die CV-Trainingsmengen überlappen; dieselben Punkte erscheinen in mehreren
Coverage-Sichten. Folds und Zusatzsichten werden nicht als unabhängige
Wiederholungen behandelt. Alle SDs und Spannweiten sind deskriptiv, keine
Konfidenzintervalle; es werden keine naiven p-Werte angegeben. Die Populationen
können sich in der AP-Verteilung unterscheiden, sodass Unterschiede von ρ allein
keinen kausalen Coverage-Effekt beweisen. Mittelwert, SD und IQR der AP je
Einzelgruppe stehen zusätzlich in `fold_metrics.csv`.

„Fehlend“ bedeutet eine unbeobachtete **diskretisierte Kombination**, nicht
zwingend einen geometrisch weit entfernten Punkt. Retrospektive Random-Trials
validieren Generalisierung innerhalb des bisherigen Suchraums; sie simulieren
nicht vollständig einen Coverage-Planner, der extreme Lücken gezielt bevorzugt.

## Phase B: vorbereitet, Training nicht ausgeführt

Mit dem vollständigen Stand von 2.000 Trials wurden pro Studie **512 neue gültige
Konfigurationen** erzeugt, über die vorhandene Lighthouse-Kandidatenlogik,
`CoverageState` und `CoverageSpec.materialize`. Optuna-Grenzen, Log-Verteilungen,
Schrittweiten und die Gültigkeit jeder Konfiguration wurden geprüft. Die festen
Suchräume enthalten keine deklarierten Bedingungen oder Constraints.

Für alle Kandidaten sind vor irgendeinem neuen Modelltraining KDE-/TPE-Scores,
Strength und individueller Coverage-Gewinn gespeichert. Historische Konfigurationen
und Duplikate wurden ausgeschlossen. Die vordefinierte Auswahl enthält je 25 Punkte:
Anker für minimale/maximale Score- und Gain-Ränge sowie Bewertungsdifferenzen,
danach maximale Mindestdistanz in diesem vierdimensionalen Rangraum.
Tie-Break: Kandidaten-ID. Es gibt keine Auswahl ausschließlich hoher Scores.
Bei XGBoost variiert der individuelle Gain von 1 bis 4; bei CatBoost ist er
in der vollständigen Vier-Parameter-Projektion stets 1. Gain-Vielfalt ist in
dieser CatBoost-Sicht daher konstruktionsbedingt nicht möglich.

Die historische Trainingskonfiguration ist in der schreibgeschützt gelesenen
MLflow-Datenbank belegt: XGBoost `device=cuda`, CatBoost `task_type=GPU`.
Hier sind keine NVIDIA-Geräte zugänglich; `nvidia-smi` beendet sich mit Code 9,
weil es den NVIDIA-Treiber nicht erreichen kann. In der DeepCAVE-Umgebung fehlen
zusätzlich beide Modellbibliotheken. Eine CPU-Ausführung wäre nicht die
vorgegebene unveränderte Trainingskonfiguration und wurde nicht ersatzweise gestartet.

| Studie | Historische Trialdauer: Median / P90 | 25 Trials: Median-/P90-basierte Hochrechnung |
|---|---:|---:|
| XGBoost | 31,8 / 67,2 s | 13,2 / 28,0 min |
| CatBoost | 55,2 / 132,1 s | 23,0 / 55,0 min |

Die Hochrechnungen sind Planungswerte, keine Laufzeitgarantien. Vorbereitet sind
maximal 25 Trials mit einem vorgeschlagenen Wandzeitbudget von 30 Minuten pro
Studie; bei CatBoost würde insbesondere der P90-basierte Aufwand das Budget
überschreiten. Tatsächlich ausgeführte neue Trainings: **null**. Ergebnisse bleiben
getrennt von den Originalstudien; Objectives der vorbereiteten Punkte sind leer,
Status `not_executed`. Der Benchmark ist kein Trainingsrunner für ein späteres
GPU-System. Dafür müssen ursprünglicher Datensatz, damalige Trainingslogik,
Splits, Seeds und Failure-Behandlung verifiziert wiederhergestellt werden.
Historische Konfiguration und Datenkennung stehen in
[phase_b_feasibility.json](phase_b_feasibility.json).

## Reproduktion und Artefakte

Aus dem DeepCAVE-Repository:

```bash
UV_CACHE_DIR=/tmp/deepcave-uv-cache PYTHONPATH=. \
MPLCONFIGDIR=/tmp/deepcave-matplotlib OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --locked python examples/api/benchmark_tpe_kde_coverage.py

UV_CACHE_DIR=/tmp/deepcave-uv-cache PYTHONPATH=. OPENBLAS_NUM_THREADS=1 \
uv run --locked python examples/api/verify_tpe_kde_coverage.py

UV_CACHE_DIR=/tmp/deepcave-uv-cache PYTHONPATH=. \
MPLCONFIGDIR=/tmp/deepcave-matplotlib OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --locked pytest tests/test_tpe_kde_coverage.py tests/test_tpe_kde_benchmark.py --no-cov -q
```

[Experimentcode](../../../../examples/api/benchmark_tpe_kde_coverage.py) ·
[Unabhängiger Ergebnischeck](../../../../examples/api/verify_tpe_kde_coverage.py)

Pro Studie enthalten die Ordner:

- `design.json`, `training_coverage.json`, `metadata.json`: outcome-freie
  Auswahlregel, Gruppengrenzen, Belegungen, Diskretisierungen, Versionen und Code-Hashes.
- `oof_scores.csv`: alle Konfigurationen, Fold-IDs, Objectives, beide Scores und Rangfehler.
- `oof_coverage.csv`: alle sechs Coverage-Sichten je Testpunkt und Gruppenzugehörigkeiten.
- `fold_metrics.csv`, `summary.csv`, `paired_metrics.csv`, `group_differences.csv`:
  Einzelmetriken, Zusammenfassungen und gepaarte Methoden-/Populationsunterschiede.
- `score_bins.csv`, `coverage_rank_errors.csv`: tatsächliche AP-Gruppenmittelwerte
  und Zusammenhang zwischen Coverage und Rangfehler.
- `fit_diagnostics.json`, `reference_checks.json`: Fit- und Bandbreitendiagnosen,
  Good-/Bad-Trial-IDs sowie Abgleich mit historischen Scores.
- `prospective_pool.csv`, `prospective_selected.csv`, `prospective_design.json`:
  neue Konfigurationen und vorab gespeicherte Scores, Auswahlreihenfolge und
  Auswahlregel; keine behaupteten neuen Objective-Werte.
- PNG- und SVG-Abbildungen der Score-Performance-Beziehung.

**34 gezielte Tests bestanden.** Der unabhängige Check zählt die Coverage über
eigene Tupel-Dictionaries nach, ohne die Gain-/Offset-Berechnung von `CoverageState`
zu verwenden: 24.000 Coverage-Zeilen, 600 Fold-/Populations-/Methodenmetriken,
2.470 Scoregruppen-Mittelwerte, 300 gepaarte Methodenvergleiche und 1.024 neue
Kandidaten einschließlich der 50 ausgewählten Punkte wurden geprüft.
Alle nachgerechneten Korrelationen und historischen Scores stimmen exakt überein.
[Prüfprotokoll](verification.json)
