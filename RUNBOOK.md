# DeepCAVE Runbook

## Plotly und Bildexport einrichten

Die getestete Kombination ist Plotly 6.1.2, Kaleido 1.4.0, Dash 2.18.2
und dash-extensions 0.1.13.
Die Versionen sind in `requirements.txt` festgelegt. Dash verwendet damit auch im
Browser das zu Plotly passende JavaScript-Bundle.

```bash
uv pip install --python .venv/bin/python -e '.[dev,optuna]'
.venv/bin/python -c 'import kaleido; kaleido.get_chrome_sync()'
```

Kaleido benötigt Chrome auch auf Servern für PNG-/PDF-Exporte. Der zweite Befehl
lädt Chrome einmalig in das Benutzerverzeichnis herunter. Ein vorhandener Chrome-
oder Chromium-Browser kann alternativ über `BROWSER_PATH` angegeben werden.
Nach dem Update laufende DeepCAVE-Server und Worker neu starten und die Seite neu laden.

Der manuelle Smoke-Test prüft die vier überarbeiteten Diagramme im Browser,
einen Slider-Callback sowie PNG- und PDF-Exporte mit synthetischen Daten:

```bash
uv pip install --python .venv/bin/python playwright
.venv/bin/python -m tests.plotting_smoke
```

Die Vorschaubilder und PDFs liegen danach unter `cache/plotting-smoke/`.

## Starten

```bash
docker compose up -d redis
uv run deepcave --open
```
