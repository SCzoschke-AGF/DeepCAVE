# Arbeitsanweisung: DeepCAVE schneller laden, unmittelbar bedienen, Planung synchron halten

## Auftrag und Ausgangspunkt

Optimiere **General**, **1-D Analyses**, **2-D Analyses** und **HPO Planner** hinsichtlich Erstaufruf, wiederholter Navigation und Reaktion auf Bedienelemente. Änderungen des geplanten Search Space müssen zwischen HPO Planner und 2-D Analyses in beide Richtungen erhalten bleiben und spätestens beim Betreten der anderen Seite wirksam sein.

Diese Anleitung beruht auf einer Codeinspektion des aktuellen Arbeitsverzeichnisses am 14.09.2026. Für ihre Erstellung wurde kein Anwendungscode verändert. Es wurden keine Browserbenchmarks und keine vollständigen Anwendungstests durchgeführt. Die unten genannten Laufzeitursachen sind aus den Ausführungspfaden abgeleitet; ihre tatsächlichen Zeitanteile musst du messen. Ein isolierter Aufruf der vorhandenen Reducer-Funktion bestätigt lediglich: Ein identischer Intent erhöht die Revision erneut, obwohl die Auswahl gleich bleibt.

Beginne mit `git status --short` und dem aktuellen Diff. Es bestehen bereits lokale Änderungen in `MANIFEST.in`, `deepcave/assets/custom.css`, `deepcave/layouts/main.py`, `deepcave/plugins/__init__.py`, `deepcave/plugins/hyperparameter/hpo_planning.py`, `deepcave/plugins/hyperparameter/two_dimensional_analysis.py`, `docs/plugins/hpo_planning.rst` und `tests/test_plugins/test_hpo_planning.py`; außerdem existiert die noch unversionierte Datei `deepcave/assets/hpo_planning_markers.js`. Baue auf diesem Arbeitsstand auf und erhalte seine Funktionen. Keine Rücksetzung bestehender Änderungen.

Die folgenden Dateipfade sind relativ zur Repository-Wurzel. Funktionsnamen sind die maßgeblichen Fundstellen. Der Stack ist laut `requirements.txt` Dash **2.18.2**, dash-extensions **0.1.13**, Plotly **6.1.2** und dash-bootstrap-components **1.0.3**. `deepcave/__init__.py` verwendet `DashProxy` mit `MultiplexerTransform`, `TriggerTransform` und `NoOutputTransform`. Entwickle gegen diese Kombination; ein Framework-Upgrade ist keine Voraussetzung.

## 1. Befunde und Prioritäten

| Priorität | Fundstelle | Aktueller Ausführungspfad und Konsequenz |
| --- | --- | --- |
| P0 | `layouts/main.py`: `mount_persistent_analysis`, `display_page` | Drei Analyseseiten werden gemeinsam eingebunden, sobald ein Run vorhanden ist, auch auf General. Das kann versteckte Analyseberechnungen auslösen und den Erstaufruf belasten. Die Sichtbarkeit wird bereits clientseitig umgeschaltet: diese Verbesserung erhalten. |
| P0 | Dieselben Funktionen | Wiederverwendung basiert auf `run_id` und `ready`. Ein Refresh mit gleicher Run-ID hat keine eigene Inhaltsrevision im Mount-Zustand. `global-run-selection-trigger` wird in `layouts/sidebar.py` aus dem Auswahlwert erzeugt, nicht explizit aus einem erfolgreich erneuerten Snapshot. Risiko veralteter Ansichten nach Refresh. |
| P0 | `plugins/hyperparameter/hpo_planning.py`: `synchronize_planner` | Eine Parameteränderung ersetzt `ROWS_ID.children` für alle Parameter und erzeugt Sidebar, Dict-Vorschau und Coverage-Figure erneut. Der Callback überträgt dazu Metadaten und Trial-Daten als `State`. Gewöhnliche Änderungen senden die gesamte Auswahl als `mode="replace"`. |
| P0 | `plugins/hyperparameter/two_dimensional_analysis.py`: `output_view_inputs`, `load_outputs` | Jede Änderung des globalen Planungsstores durchläuft den allgemeinen DynamicPlugin-Callback. `process()` kann aus dem Cache kommen, aber JSON wird erneut gelesen, DataFrames werden aufgebaut, Trials sortiert und alle drei Figures neu erzeugt. Für Planungsänderungen ändern sich im Wesentlichen die Masken. |
| P0 | `layouts/main.py`: `update_planned_search_space`; `hyperparameter/search_space_state.py`: `apply_planning_intent` | Gemeinsamer Store und Run-Trennung sind vorhanden. Identische Intents erhöhen trotzdem `revision`; es fehlen Ereignisdeduplizierung und Schutz gegen veraltete Zustände. Ein vollständiger HPO-Ersatz kann zwischenzeitliche Änderungen anderer Parameter überschreiben. Das konkrete Verhalten überlappender Browsercallbacks ist zu testen. |
| P0 | Dieselben Callbacks, `HPOPlanning.register_callbacks` | Deklarierter Pfad: globaler Store → HPO-Synchronisierung → HPO-Intent → globaler Store. Zusätzlich gibt es dynamisch erzeugte Controls als Rückweg. Callback-Zyklen und Lebensdauer fehlender Komponenten müssen unter dem tatsächlichen DashProxy-Setup geprüft und strukturell bereinigt werden. |
| P1 | `layouts/general.py`: `__call__`, `_callback_working_directory_changed`, `_callback_discover_optuna_studies`, `_callback_update_optuna_button` | General wird bei Rückkehr über `display_page` neu erzeugt. Der Verzeichnis-Callback hängt auch an der URL; das Working-Directory-Feld hat kein Debouncing. Sowohl Study-Discovery als auch Button-Zustand rufen `list_optuna_studies()` auf. Wiederholte Dateisystem-/SQLite-Zugriffe vermeiden. |
| P1 | `plugins/__init__.py`: `plugin_input_update`; `plugins/dynamic.py`: `plugin_output_update` | Viele Controls einschließlich Optionslisten hängen an allgemeinen Callbacks. Abhängigkeiten werden neu aufgelöst und komplette Inputlisten zurückgeschrieben. Der bestehende URL-Guard verhindert bereits einen Teil der Navigationsupdates. `previous_inputs` und `raw_outputs` liegen auf gemeinsam verwendeten Plugin-Instanzen. |
| P1 | `hyperparameter/analysis_utils.py`: `_get_trial_table`; `utils/run_caches.py`: `get` | Gemeinsamer Trial-Tabellen-Cache für 1-D/2-D ist bereits vorhanden: acht Einträge, Run-/Versions-/Objective-/Richtungsbezug. Die Tabellenerstellung erfolgt unter einem globalen Lock. Process-Ergebnisse werden dagegen bei jedem `rc.get()` aus JSON-Dateien gelesen. |
| P1 | `hyperparameter/hpo_planning.py`: `process`, `_parameter_insights`, `_summarize_features` | Planner scannt die Historie separat, rekonstruiert Configurations und berechnet Insights für alle Parameter vor deren Anzeige. Coverage ist bereits auf höchstens 100 Top-Trials begrenzt; deren kleine Größe rechtfertigt lokale UI-Berechnungen. |
| P2 | `layouts/main.py`, `layouts/sidebar.py`, `config.py` | `global-update` läuft standardmäßig alle 500 ms. Run-Auswahl vermeidet schon identische Outputs; Queue-Informationen werden weiterhin regelmäßig abgefragt und aufgebaut. Das ist zusätzliche Arbeit auch ohne aktive Berechnung. |
| P2 | 1-D/2-D `load_outputs`, `utils/styled_plotty.py`: `save_image` | Je drei PDF-Exportaufrufe pro Darstellung. **Standardmäßig ist `SAVE_IMAGES=False`**, dann endet der Helper sofort. Nur bei aktivierter Option entsteht synchroner Exportaufwand; nicht pauschal als Ursache der normalen Latenz darstellen. |

## 2. Zuerst messbar machen

Erstelle eine reproduzierbare Browsermessung mit echten Komponenten und der tatsächlichen Callback-Registrierung. Nutze synthetische Optuna-Runs, damit keine produktiven Studien verändert oder ihre Caches gelöscht werden. Prüfe zusätzlich eine repräsentative vorhandene Studie ausschließlich lesend, falls verfügbar.

Erfasse getrennt:

- Vollständig kalter Start mit isoliertem leerem Cache, frischer Browser bei warmem Servercache, erster Besuch jeder Seite und wiederholter Besuch bereits geladener Seiten.
- Reihenfolge General → 1-D → 2-D → HPO → 2-D → General sowie Direkteinstieg über jede Plugin-URL; mindestens 20 warme Wechsel pro Richtung.
- Zeit vom Eingabeereignis bis zur ersten sichtbaren Rückmeldung und bis zum korrekten stabilen Ergebnis. Miss nicht nur HTTP-Antwortzeiten, sondern auch Plotly-Rendering und Browser-Hauptthread.
- Callback-Anzahl, Request-/Response-Größen, `process()`-/`load_outputs()`-Aufrufe, Cache-Treffer, Dateizugriffe, Browser-DOM-Umfang und Speicher über wiederholte Wechsel.
- Zeitanteile von Trial-Aufbereitung, Aggregation, Cache-Deserialisierung, Figure-Erstellung und optionalem Bildexport, zum Beispiel mit `time.perf_counter()` und Browser-Performance-Marks.

Verwende eine kleine Studie, eine Referenzstudie mit etwa 10.000 Trials und 20 gemischten Parametern sowie eine größere Skalierungsstudie, beispielsweise 100.000 Trials. Berücksichtige lineare/logarithmische Zahlen, Integer-Schritte, Kategorien, bedingte Parameter und Gruppen. Dokumentiere Hardware, Browser, Workerzahl, Datenumfang und `SAVE_IMAGES`.

Arbeitsziele für die Referenzstudie auf derselben Maschine, **keine bereits gemessenen Zusagen**:

| Vorgang | Ziel |
| --- | --- |
| Bereits geladene Seite anzeigen | p95 ≤ 100 ms; keine neue Datenaufbereitung und kein Figure-Neubau durch Navigation allein |
| Drag-/Button-Rückmeldung | p95 ≤ 50 ms; keine Netzwerkabhängigkeit für rein lokale Darstellung |
| Search-Space-Commit einschließlich sichtbarer abhängiger Ansicht | p95 ≤ 150 ms; kein erneutes `process()` |
| Erstaufruf | Sichtbare Seite zuerst bedienbar; kalte/warm gestartete Messungen getrennt; gegenüber Ausgangswert keine Regression, messbare Senkung des kritischen Pfads |
| Längere echte Berechnung/Import | Sofort sichtbarer laufender Zustand; UI bleibt bedienbar; Abschlusszeit separat ausweisen |

Sind Zeitziele unter den dokumentierten Bedingungen nicht erreichbar, liefere Messwerte und konkrete verbleibende Ursachen. Verkürze keine Analysen durch stilles Weglassen von Trials.

## 3. Gemeinsamen Planungszustand zuerst korrekt und schnell machen

### 3.1 Zustandsvertrag

Behalte `planned-search-space-state` als **eine kanonische, im Hauptlayout verankerte Browserquelle**. Lokale Auswahlstores sind nur abgeleitete Ansichten oder kurzfristige Drag-Vorschauen. Der aktuelle `storage_type="memory"` gilt weiter: Navigation erhält den Zustand, ein vollständiger Reload setzt ihn zurück. Persistenz über Reloads ist nicht verlangt und darf nicht nebenbei eingeführt werden.

Erhalte und teste folgende Regeln:

- Planung ist je Run/Gruppe und Parametername getrennt. Objective, Elite-Anteil, Parameterpaar, Bins und Route ändern keine geplanten Grenzen.
- Gewöhnliche Änderungen patchen genau den betroffenen Parameter. Auch HPO darf hierfür nicht mehr die komplette Auswahl ersetzen.
- HPO-Reset setzt alle Parameter des aktiven Runs zurück; 2-D-Reset nur die zwei ausgewählten Parameter. Andere Runs und Parameter bleiben erhalten.
- Kanonische Werte bleiben Originalwerte mit erhaltenen Typen, nicht Sliderkoordinaten. Log-Skalierung, Integer-/Step-Snapping, Kategorienreihenfolge, Ordinals, Konstanten und erlaubte Erweiterungen über den bisherigen Raum hinaus bleiben fachlich gleich.
- Fehlende/inaktive Parameter bleiben korrekt behandelt. Leere Kategorienauswahl nicht still in „alle“ verwandeln. Ungültige Eingaben einschließlich NaN/Infinity und nichtpositiver Log-Werte dürfen den gültigen Store nicht beschädigen.
- Planung mutiert weder den ursprünglichen `run.configspace` noch bestehende Trials. 2-D maskiert die geplanten Bereiche; seine empirischen Statistiken werden dadurch nicht auf den Ausschnitt neu definiert.

### 3.2 Einen eindeutigen Schreibpfad implementieren

Bevorzuge einen **synchronen clientseitigen Reducer** für die kleinen Planungsereignisse. So wird die Änderung im Browser übernommen, bevor eine Serverantwort oder ein Seitenwechsel sie überholen kann. Die vorhandenen Python-Helfer dienen als fachliche Referenz; ergänze gemeinsame Testvektoren für Python und JavaScript, besonders bei Rundung (`round()` und `Math.round()` sind nicht überall gleich), Logarithmen und typisierten Kategorien.

Wichtig: Die bestehende Store→Callback→Intent→Store-Kette nicht einfach in mehrere JavaScript-Callbacks übersetzen. Dash unterstützt Synchronisierung innerhalb eines Callbacks, aber keine zirkulären Ketten über mehrere Callbacks. Plane deshalb die Eigentümer der Properties vor der Umsetzung. [Dash: Circular Callbacks](https://dash.plotly.com/advanced-callbacks#circular-callbacks)

Empfohlene konkrete Struktur:

1. Stabile Root-Stores enthalten aktiven Datenkontext und kleine Planungsmetadaten. Struktur-/Metadatenupdates hängen von Run-Version und Parameterauswahl ab, nicht von jeder Planungsänderung.
2. Ein zentraler clientseitiger Callback verarbeitet die Controls beider Seiten (Pattern-Matching mit `ALL` für auch leere Komponentenlisten), Reset-Ereignisse und Metadaten-/Mount-Ereignisse. Er liest den bisherigen kanonischen Store als `State` und schreibt sowohl den neuen Store als auch die synchronisierten Control-Properties. Eine notwendige Control-Wert-Rückkopplung bleibt damit innerhalb dieses einen Callbacks. Entferne konkurrierende bisherige Writer.
3. Reine Ansichts-Callbacks lesen die Auswahl und schreiben Masken, Sidebar, Marker oder Coverage. Sie schreiben weder Intents noch kanonische Auswahl noch Eingabe-Werte zurück. Strukturcallbacks dürfen keinen indirekten Rückkopplungspfad herstellen.
4. Hydrierung neu eingebundener Controls hat Vorrang vor deren Defaultwerten. Ein Mount-/Metadatenereignis liest die gespeicherte Auswahl und darf niemals Defaultwerte als Benutzeränderung committen. Synchronisiere Struktur und Metadaten mit einem gemeinsamen Kontext; teste diese Reihenfolge im Browser.

Das ist eine Zielstruktur, kein ungeprüft kopierbares Callback-Snippet: überprüfe Pattern-Matching, dynamische Komponenten und die DashProxy-Transforms mit einem kleinen Integrationstest. Falls ein anderer Aufbau nötig ist, muss er dieselben Eigenschaften erfüllen: genau ein Store-Writer, kein deklarierter Callback-Zyklus, eindeutige Property-Eigentümer und keine Roundtrips für den Commit. `allow_duplicate` oder der Multiplexer ersetzen keine Konfliktauflösung.

Jedes Ereignis trägt den **zum Control gehörenden** Run-/Metadatenkontext und Parameternamen. Nicht erst nachträglich anhand des inzwischen geänderten `global-run-selector` entscheiden, wohin ein alter Sliderwert gehört. Veraltete Metadaten-/Datenversionen ablehnen; gegen derzeitige Metadaten validieren. Falls asynchrone Ereignisse oder Ergebnisse vorkommen, ergänze Ereignis-ID/Sequenz und Versionsprüfung. Eine Revision allein verhindert kein Überschreiben eines neueren Ergebnisses.

Idempotenz: Normalisieren, zusammenführen, vergleichen; ohne semantische Änderung `no_update`, keine Revisionserhöhung und kein Folge-Rendering. Bei mehreren geänderten Inputs alle relevanten Ereignisse geordnet behandeln, statt ausschließlich `ctx.triggered[0]` zu verwenden. Änderungen verschiedener Parameter müssen sich addieren; für denselben Parameter gilt die jüngste gültige Benutzeraktion. Lokale Vorschauen und programmgesteuertes Hydrieren sind keine neuen Benutzeraktionen.

### 3.3 Drag-Vorschau und Commit trennen

`RangeSlider` verwendet in HPO und 2-D bereits `updatemode="mouseup"`. Erhalte das für teure Folgearbeit. Verwende `drag_value` für unmittelbare lokale Vorschau von Zahlen, Auswahlflächen und gegebenenfalls Masken; `value` committed beim Loslassen. Beide Properties sind in den lokal installierten Dash-Komponenten vorhanden.

Keine Serveranfrage pro Drag-Schritt. Falls eine Vorschau selbst teuer ist, auf Animationsframes begrenzen und nur die sichtbare Ansicht zeichnen. Während des Drags keine Slider-Domain ändern oder Zeilen ersetzen: das kann die Position unter dem Mauszeiger verschieben. Bei Commit Grenzen, Tooltips und Domain gemeinsam konsistent aktualisieren. Schnelle Tastenfolgen bzw. numerische Eingaben dürfen Zwischenstände zusammenfassen, aber den letzten gültigen Wert nicht verlieren.

## 4. Seitenlebensdauer und Dateninvalidierung

Überarbeite `layouts/main.py`, `layouts/general.py`, `layouts/sidebar.py` und die betroffenen Plugin-Lebenszyklen gemeinsam.

**General dauerhaft einbinden.** Gib General einen eigenen stabilen Container und schalte seine Sichtbarkeit clientseitig wie bei den Analyseseiten. Ein Hin-/Zurückwechsel darf seine Komponenten, Study-Auswahl oder Gruppenfelder nicht neu initialisieren. Normale Navigation darf keinen Verzeichnisscan auslösen. Tatsächliche Änderungen an Runs/Gruppen/Working Directory aktualisieren gezielt die betroffenen General-Ansichten.

**Sichtbare Analyse zuerst, weitere bei Bedarf.** Ersetze das pauschale Einbinden aller drei Analysen durch ein kleines Mount-Verzeichnis pro Seite. Erstbesuch bindet nur die benötigte Seite ein; bereits geladene Seiten bleiben für den aktiven Datenkontext bestehen. Ergänze neue Seiten, ohne den gemeinsamen `children`-Baum samt bestehenden Seiten erneut auszugeben. Optionales Vorladen erst nach sichtbarer Fertigstellung und mit niedriger Priorität, dedupliziert und begrenzt. Nicht pauschal alle Parameterpaare vorausberechnen.

Unterscheide `mounted`, `data_ready`, `dirty`, `error` und den Datenkontext; ein erfolgreich erzeugtes Layout bedeutet noch nicht, dass Figures fertig sind. Reines Ausblenden beendet Dash-Callbacks nicht. Teure inaktive Ansichten sollen neue Analyseanforderungen nur als „dirty“ merken und beim Aktivieren einmal aktualisieren. Gemeinsame Planungs-Commits laufen unabhängig von der Sichtbarkeit weiter. Eine inaktive Ansicht darf entweder ihre kleinen Ableitungen sofort aktualisieren oder beim Aktivieren die jüngste Revision übernehmen; veraltete Werte dürfen beim Betreten nicht kurz als gültige Auswahl erscheinen.

**Explizite Datenrevision hinzufügen.** Nach erfolgreichem Import/Refresh, Run-Entfernung, Gruppenänderung und Cache-Löschung ein gezieltes Kontext-/Invalidierungsereignis auslösen. Verwende `(run_id, data_revision)` für Daten-/Layoutgültigkeit; Objective und Richtung zusätzlich für Analysen. Ein Refresh derselben Studie muss alle betroffenen Datenableitungen und Objective-Optionen aktualisieren, ohne einen künstlichen Run-Wechsel zu benötigen.

Trenne dabei:

- Datenrevision: neue Trials/Objectives → analytische Caches ungültig.
- Planungsmetadaten-Version/Fingerprint: Parameterdefinition geändert → gespeicherte Auswahl mit aktuellen Metadaten abgleichen.
- Planungsrevision: neue Grenzen → nur Auswahlansichten/Masken ändern.

Erhalte gültige Planung beim Study-Refresh und Objective-Wechsel. Bei geänderten Typen, Kategorien oder Step-Werten gezielt reconciliieren; entfernte Parameter nicht wieder einschleusen. „Clear Plugin Caches“ leert auch neu eingeführte analytische Speicher-Caches und markiert Ansichten ungültig, löscht aber nicht ungefragt die Benutzerplanung. Run A → B → A stellt die Planung von A wieder her.

Jede asynchrone Ergebnisübernahme prüft Run, Datenrevision, Analyse-Key und gegebenenfalls die relevante Planungsrevision. Alte Ergebnisse dürfen weder neue Figures überschreiben noch frische Planung auf alte Werte zurücksetzen. Vermeide durch Run-Remount und parallele alte Inputcallbacks doppelt gestartete Berechnungen.

Behebe die leeren Zustände: `mount_persistent_analysis` gibt bei fehlendem Run/Worker derzeit keine Seite zurück, während die Route „Loading analysis …“ zeigen kann. Ohne passenden Run, bei inkompatiblen Daten und Fehlern eine klare abgeschlossene Anzeige liefern. Die drei fokussierten Plugins sind `DynamicPlugin` und berechnen synchron; prüfe, ob ihre UI tatsächlich `queue.ready()` benötigt. Etwaige Entkopplung eng auf diese Seiten begrenzen, StaticPlugin-Workeranforderungen erhalten.

Beim Sichtbarwerden nur die betroffenen Graphen passend skalieren; bestehende Client-Navigation und Plotly-Zoom/Legendenzustände erhalten. Inaktive Seiten dürfen keine unbeschränkte DOM-/Speicheransammlung erzeugen. Persistenz aller Layouts über beliebig viele Runs ist nicht erforderlich.

## 5. Ansichten nach tatsächlichen Abhängigkeiten aufteilen

| Ereignis | Erforderliche Arbeit | Zu vermeiden |
| --- | --- | --- |
| Navigation zu gültiger geladener Seite | Sichtbarkeit, gegebenenfalls kleine ausstehende Planungsableitung, Resize | `process`, Datei-/DB-Zugriff, kompletter Figure-Neubau |
| Search-Space-Änderung | Kanonischer Commit, betroffene Controls, Masken, Auswahlsummary, Coverage | Trial-Aufbereitung, Binning, Ranking, alle Parameterzeilen ersetzen |
| 2-D „Color by rank“ | Farbe, Farbskala/Farbbar und zugehörige Hoverdarstellung des Scatterplots | Enrichment-/Median-Figure neu erzeugen |
| 2-D Bin-Zahl | Betroffene Bin-Zuordnung und gemeinsame Aggregation, binned Figures/Maskengeometrie | Historie neu einlesen, unveränderten Rohdaten-Scatter neu aufbauen |
| Elite-Anteil | Elite-abhängige Statistiken und Darstellungen | Unveränderte Trial-/Basisdaten erneut laden |
| Parameter/Parameterpaar | Passende Datenprojektion, Aggregation, Figures und Planungscontrols aus vorhandenem Store | Neue Trial-Gesamttabelle je Auswahl |
| Objective/Richtung oder Datenrevision | Davon abhängige Rankings/Analysen invalidieren | Planung zurücksetzen |

### 5.1 2-D Analyses

Entkopple den globalen Planungsstore vom allgemeinen `DynamicPlugin.plugin_output_update`. Das bestehende `output_view_inputs` verhindert nur die Aufnahme ins Process-Cache-Key, nicht den teuren Darstellungsdurchlauf.

Erzeuge Basis-Figures nur bei geänderten Analyseinputs. Leite die Masken aus kleiner Achsengeometrie, Parameterdefinitionen und der Auswahl ab. `_apply_rectangular_mask`, `_apply_continuous_mask`, `_apply_scatter_mask` und `_apply_binned_mask` enthalten die vorhandene fachliche Geometrie; erhalte ihre Sonderfälle, insbesondere exakte logarithmische Bin-Grenzen, Kategorien, Lücken und überlappungsfreie Maskenecken.

Ändere bei einer Planungsaktion nur die entsprechenden `layout.shapes` bzw. tatsächlich abhängigen Darstellungsattribute. `dash.Patch` ermöglicht partielle Updates seit Dash 2.9 und reduziert vollständige Figure-Übertragungen; eine Patch-Antwort beseitigt aber nicht automatisch vorherige Python-Arbeit. Für die unmittelbare Vorschau ist eine clientseitige Ableitung aus kleiner Geometrie vorzuziehen. [Dash: Partial Property Updates](https://dash.plotly.com/partial-properties)

Vermeide konkurrierende Figure-Writer für vollständige Neuberechnung und Masken. Bevorzugt liefert der Server eine versionierte Basis, und ein einziger Darstellungs-Callback verbindet diese mit der aktuellen Auswahl. Wenn serverseitige Patches verwendet werden, muss ihre Basisversion überprüft werden. Eine Änderung an einem dritten, nicht angezeigten Parameter oder einem anderen Run bewirkt in 2-D `no_update`.

Nutze stabile `uirevision`-Schlüssel: Planungs-/Darstellungsänderungen erhalten Zoom und Pan; Run-/Achsenwechsel dürfen sie gezielt zurücksetzen. Bins und Color-Schalter werden unabhängig behandelt. Cache-Hits dürfen nicht mehr sämtliche Figures neu aufbauen.

### 5.2 HPO Planner

Zerlege `synchronize_planner` entlang von Zustandsänderung und Darstellung. `_planner_rows()` soll nur beim Aufbau bzw. einer echten Änderung der Parameterstruktur alle Zeilen erzeugen. Normale Änderungen schreiben Eigenschaften des betroffenen Controls und seiner Overlays; Sidebar und Coverage dürfen separat aktualisiert werden. Erhalte Fokus, laufenden Drag, Marker-Schalter und Insight-Auswahl.

Das vorhandene `hpo_planning_markers.js` macht Top-Trial-Schalter bereits clientseitig. Erhalte dies und binde eine Änderung der Auswahl/Domain korrekt in die Markerpositionierung ein. Aktuell kaschiert der Zeilenneubau diese Aktualisierung; fällt er weg, dürfen Marker nicht auf alten Positionen bleiben. DOM-Manipulationen außerhalb des Dash-/React-Zustands vermeiden.

Coverage arbeitet auf maximal 100 Trials. Berechne Parameterverletzungen, Badge und reine Anzeigeänderungen möglichst im Browser aus dem bereits vorhandenen kleinen Datensatz. Bei einer Bereichsänderung bleiben Trial-Ränge und Objective-Koordinaten gleich; aktualisiere vor allem Status/Farbe/Hover. Aufwendige Spread-Daten nicht bei jeder Slideraktion vom Browser zurück zum Server übertragen. Falls sehr lange Spread-Listen vorkommen, Statistik vorab berechnen und Detailpunkte bedarfsgerecht laden; keine stille Änderung der vorhandenen Scatter-Funktion.

Entkopple Insight-Panel öffnen/schließen und Elite-Anzeigewechsel von der Gesamtplanung. Prüfe den Zeitanteil von `_parameter_insights()` beim Erstaufruf. Falls dominant: Insights pro Parameter bedarfsgerecht berechnen und cachen; das Panel sofort öffnen, den Inhalt gezielt liefern, weitere Daten erst im Leerlauf vorbereiten. Summaries, die zur sichtbaren Planung gehören, bleiben korrekt verfügbar.

### 5.3 1-D Analyses

Erhalte den vorhandenen gemeinsamen Trial-Tabellen-Cache. Trenne möglichst Basisprojektion/Bin-Zuordnung von Elite-abhängigen Aggregaten. Bei unverändertem Parameter und Objective benötigt eine Elite-Änderung keine neue Rohdaten-Scatter-Figure; aktualisiere nur tatsächlich betroffene Panels und Rugs. Beim Rückwechsel auf einen bekannten Parameter passende Ergebnisse wiederverwenden.

Verhindere, dass identische Optionslisten und Werte neue Berechnungen auslösen. Reines `filter=True` genügt nur für echte Anzeigeparameter; Elite-Anteil und Bin-Zahl dürfen nicht aus dem Daten-Key entfernt werden, solange `process()` davon abhängt. Eingaben vor Verarbeitung validieren und kurzzeitig unvollständige Zahlen ohne Fehlerkaskaden behandeln. Keine zusätzliche Apply-Schaltfläche einführen.

### 5.4 General

Entferne reine URL-Wechsel als Anlass für Discovery. Debounce das Working-Directory-Textfeld; Verzeichnisbuttons bleiben direkt wirksam. Fasse Study-Discovery in einem kleinen Ergebnis-/Metadatenstore zusammen, aus dem Dropdown und Load-/Refresh-Button lesen. Der Button-Callback muss nicht erneut alle Studies abfragen; unmittelbar vor einem tatsächlichen Import weiterhin gegen die aktuelle Quelle validieren.

Falls Discovery gecacht wird: absoluter Pfad, Änderungsindikatoren und explizite Aktualisierung beachten; SQLite-WAL kann sich ändern, ohne dass allein die Hauptdatei einen ausreichenden Schlüssel liefert. Vermeide einen dauerhaft veralteten Verzeichnis-/Study-Cache.

Load/Refresh sowie aufwendiges Gruppenbilden sind echte Datenarbeit. Zeige den laufenden Zustand sofort, verhindere Mehrfachausführung und aktualisiere danach über das oben definierte Datenereignis. Nach Messung gegebenenfalls einen deduplizierten Hintergrundjob einsetzen. Dafür Status, Fehler und Versionsprüfung implementieren; keinen zusätzlichen langen Callback-Pollingpfad für triviale UI-Aktionen einführen.

`_refresh_groups` ist aktuell Instanzzustand, der beim Layoutaufbau gesetzt wird. Ersetze diese implizite Initialisierungssperre durch einen expliziten Mount-/Hydrierungskontext. Veränderte Run-/Gruppenlisten müssen General auch bei dauerhafter Einbindung erreichen.

## 6. Gemeinsame Infrastruktur gezielt verbessern

In `Plugin.plugin_input_update` nur veränderte Properties zurückgeben (`no_update` für den Rest). Erzeuge nach Abhängigkeitsauflösung einen konsistenten effektiven Analyse-Key; verhindere doppelte Berechnung für Zwischenkombinationen von Run, Objective und Parametern. Die bestehenden URL- und globalen Objective-Regeln erhalten und testen.

Nutze bei Bedarf einen begrenzten Speicher-Cache oberhalb von `RunCaches.get()`. Schlüssel: Run-ID, Datenrevision, Plugin-/Schemasversion, Objective samt Richtung und nur tatsächlich relevante Analyseinputs. Verzeichnispfade/Optionslisten, Route und Planungsrevision gehören nicht in einen reinen Analysedaten-Key. Eine Schemaänderung invalidiert auch ältere persistierte Resultate; HPO und 2-D haben bereits eigene Cache-Schema-Versionen.

Keine global geteilten veränderlichen Figures nachträglich mit benutzerspezifischen Masken beschreiben. Rohdaten/Cacheeinträge unveränderlich behandeln oder beim Ableiten gezielt kopieren. UI-Auswahl, `previous_inputs` und benutzerspezifische `raw_outputs` dürfen nicht durch andere Browserkontexte überschrieben werden. Betroffene Callbackpfade auf Browser-/Run-Kontext umstellen; eine vollständige Multiuser-Neuarchitektur ist hierfür nicht verlangt.

Der Trial-Tabellen-Cache verwendet derzeit `id(run)`, `latest_change`, Historienlänge, Objective und Richtung. Führe neue Revisionen konsistent ein; Hashing kompletter Optuna-Pickles gehört nicht in Slider-/Navigationscallbacks. Vollständige Inhaltsprüfung an expliziten Lade-/Refreshpunkten erhalten. Gruppen müssen Revisionen ihrer Mitglieder berücksichtigen.

Verhindere doppelte Berechnung desselben Keys („single flight“). Wenn der globale Tabellen-Lock als Engpass gemessen wird, pro Key koordinieren, statt während eines großen Tabellenaufbaus unabhängige Runs zu blockieren. Speicher nach Größe und/oder Eintragszahl begrenzen; veraltete Versionen entfernen. Ein Prozess-LRU wirkt nur in seinem Prozess; keine workerübergreifende Wirkung behaupten.

HPO kann gemeinsame Konfigurations-/Trial-Aufbereitung nutzen, muss aber seine Semantik behalten: User-Attribute als Objective, Spread-Attribute, bedingte Parameter, wiederholte Konfigurationen und vollständige Historie für Summaries/Insights. Nicht die auf 100 begrenzten Coverage-Trials zur Grundlage aller Statistiken machen. Keine O(P²)-Vorberechnung sämtlicher Parameterpaare.

Reduziere Hintergrundlast in `layouts/sidebar.py`: unveränderte Queue-Ansichten mit `no_update`, Run-/Gruppenänderungen über Ereignisse; Leerlauf-Polling nur bei Bedarf bzw. langsamer. `REFRESH_RATE` nicht pauschal hochsetzen, weil das Statusverhalten anderer StaticPlugins beeinflusst würde.

Bei `SAVE_IMAGES=True` Export aus dem interaktiven Callback-Pfad lösen oder explizit anfordern. Bestehende Downloadmöglichkeiten erhalten; asynchrone Exportaufträge deduplizieren und eindeutige Run-/Analyse-Dateinamen verwenden. Bei Standardkonfiguration hier keinen großen Gewinn unterstellen.

Clientside-Callbacks ersparen Netzwerk-/Callbackkettenkosten, können aber selbst den Browser blockieren. Kleine Metadaten und Top-Trial-Daten eignen sich dafür; vollständige Historiendatensätze müssen nicht für jede Aktion in `dcc.Store` kopiert werden. [Dash: Clientside Callbacks](https://dash.plotly.com/clientside-callbacks)

## 7. Regressionstests und Abnahme

Erweitere bestehende Tests, statt ihre fachlichen Erwartungen an eine bequemere Implementierung anzupassen:

- `tests/test_plugins/test_search_space_state.py`: No-op ohne neue Revision, Wiederholung desselben Ereignisses, unterschiedliche Parameter ohne verlorene Updates, veraltete Ereignisse, beide Reset-Reichweiten, Run-Trennung und Reconciliation nach Metadatenänderung.
- `tests/test_plugins/test_hpo_planning.py`: Fokus-/Zeilenerhalt, Integer-/Log-/Step-Parität, Marker nach Bereichsänderung, Coverage, User-Attribute, Insight-Öffnen ohne Gesamtneuberechnung, kein `replace` für Einzeländerungen.
- `tests/test_plugins/test_two_dimensional_analysis.py`: Maskengeometrie weiter exakt; Planung ruft weder `process()` noch Basis-Figure-Erstellung auf; Änderungen anderer Parameter bewirken kein Rendering; Bin-/Color-/Elite-Abhängigkeiten getrennt.
- `tests/test_plugins/test_one_dimensional_analysis.py`: korrekte Elite-Statistiken und Rugs; wiederverwendbare Basisdaten; kein neuer Scatter durch reine Elite-Änderung.
- `tests/test_layouts/test_general.py`, `tests/test_layouts/test_sidebar.py` und neue MainLayout-Integrationstests: General bleibt bestehen; keine Discovery durch Navigation; Mount der aktiven Seite zuerst; keine Remounts warmer Seiten; Refresh derselben Run-ID, Cache-Clear, Gruppenänderungen und leere Zustände.
- Bestehende `test_global_objective.py`, `test_global_run_selection.py` und Cache-/Run-Handler-Tests sichern das Zusammenspiel über die vier Seiten hinaus ab.

Füge einen Browsertest mit tatsächlichem MainLayout hinzu. Die vorhandene `tests/plotting_smoke.py` ist ein hilfreicher Darstellungs-/Exporttest, deckt jedoch diesen vollständigen Navigations-/Synchronisierungsablauf nicht ab. Ein reiner Test von `load_outputs()` belegt weder geringe Latenz noch korrekte Callback-Verkettung.

Verbindliche Browserabläufe:

1. Auf General starten, 1-D/2-D/HPO erstmals und wiederholt besuchen; Controls, Scroll-/Zoomzustand und vollständige Darstellung prüfen. Direkteinstieg in beide Planungsseiten funktioniert auch, solange die jeweils andere Seite noch nie eingebunden war.
2. In HPO numerischen/logarithmischen/kategorischen Parameter ändern, **unmittelbar** zu 2-D wechseln: Controls und Masken zeigen die Auswahl. Dort ändern, sofort zurück: HPO zeigt dieselben Originalwerte samt Dict-Vorschau und Coverage.
3. In 2-D das Parameterpaar mehrfach wechseln: Planung bleibt pro Parameter erhalten. HPO ändert einen dritten Parameter: aktuelles 2-D-Paar bleibt unverändert und wird nicht teuer neu gezeichnet.
4. Schnell mehrere verschiedene Parameter ändern und mehrfach navigieren. Künstlich verzögerte Serverantworten dürfen keine Änderung verlieren oder jüngere Ergebnisse überschreiben. Während laufender Analyse Run/Objective wechseln und danach erneut editieren.
5. Run A → B → A; beide Reset-Arten; Refresh von A bei gleicher ID mit neuen Trials; Änderung der ConfigSpace-Metadaten; Cache-Clear. Auswahl und Datenaktualität getrennt prüfen.
6. Fehlender/entfernter/inkompatibler Run, kein Worker und Fehler im Import: kein endloses Loading, keine fehlenden Callback-Komponenten, keine Wiederholungsschleifen. Zwei getrennte Browserkontexte dürfen ihre Planung nicht gegenseitig beeinflussen.
7. Konsole und `/_dash-dependencies` bzw. tatsächlichen Callbackgraphen prüfen: keine Zyklen, Duplicate-ID-Probleme oder fehlenden festen Input-Komponenten. Nach einer abgeschlossenen Interaktion muss die Anwendung wieder zur Ruhe kommen.

Für schnelle Regressionen zunächst die betroffenen Tests ausführen, zum Beispiel:

```bash
.venv/bin/python -m pytest --no-cov \
  tests/test_plugins/test_search_space_state.py \
  tests/test_plugins/test_one_dimensional_analysis.py \
  tests/test_plugins/test_two_dimensional_analysis.py \
  tests/test_plugins/test_hpo_planning.py \
  tests/test_plugins/test_global_objective.py \
  tests/test_plugins/test_global_run_selection.py \
  tests/test_layouts/test_general.py \
  tests/test_layouts/test_sidebar.py \
  tests/test_utils/test_cache.py \
  tests/test_runs/test_handler.py
```

Ergänze neu angelegte Tests explizit. Anschließend bei Änderungen an Plugin-Basis, Run-Handler oder gemeinsamem Cache die vollständige Suite und passende Format-/Typprüfungen ausführen. Beachte, dass manche `make check`-Ziele Fehler mit `|| :` verschlucken; prüfe die tatsächlichen Werkzeugausgaben. Browser- und Zeitmessungen ohne Coverage-Overhead durchführen. Setup-Hinweise stehen in `RUNBOOK.md`.

## 8. Umsetzungsreihenfolge und fertiges Ergebnis

1. Ausgangsmessung und Tests für Synchronisierung, Callbackgraph und Refresh derselben ID ergänzen.
2. Kanonischen Planungszustand, Parameter-Patches, Idempotenz und eindeutige Property-Writer herstellen.
3. Planungs-Commits und lokale Vorschauen von Analyseberechnung entkoppeln; 2-D-Masken und HPO-Zeilen gezielt aktualisieren.
4. General erhalten, sichtbare Seite bevorzugt laden, versteckte Berechnungen begrenzen und Datenrevisionen vollständig verdrahten.
5. Erst anhand der verbleibenden Messwerte Aggregations-/Cache-, Insight-, Discovery- und Pollingoptimierungen umsetzen.
6. Fachliche Regressionen, Browserabläufe und identische Vorher-/Nachher-Messungen abschließen.

Liefere mit der Umsetzung eine kurze Beschreibung des endgültigen Zustandsflusses, die geänderten Dateien, Testergebnisse und eine Tabelle mit Vorher-/Nachherwerten pro Seite und Interaktionsart. Kennzeichne ungeprüfte Punkte und verbleibende Grenzen ausdrücklich. Fertig ist die Arbeit, wenn schnelle Navigation und Eingaberückmeldung **zusammen mit** verlustfreier gegenseitiger Planung und zuverlässiger Dateninvalidierung nachgewiesen sind.
