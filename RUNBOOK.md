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

### Docker Compose (dauerhafter Remote-Dienst)

```bash
docker compose up -d --build
docker compose ps
```

DeepCAVE ist danach unter `http://<Server-IP>:8050/` erreichbar. Compose startet
Webserver, Worker und Redis gemeinsam. Beide Container nutzen `restart:
unless-stopped` und starten nach einem Host-Neustart wieder, sofern der Docker-Dienst
beim Booten aktiviert ist. Mit `docker compose logs -f deepcave` lassen sich die
Startmeldungen prüfen. Zum Beenden: `docker compose down`.

Der Webport wird standardmäßig auf allen Host-Adressen veröffentlicht. Ein anderer
Hostport lässt sich mit `DEEPCAVE_PORT=18050 docker compose up -d --build` wählen.
Für einen vorgeschalteten Reverse Proxy kann die Bind-Adresse auf localhost begrenzt
werden: `DEEPCAVE_BIND_ADDRESS=127.0.0.1 docker compose up -d --build`.
Für dauerhaft gesetzte Werte eine `.env`-Datei im Projektverzeichnis verwenden,
zum Beispiel mit `DEEPCAVE_PORT=18050` und `DEEPCAVE_BIND_ADDRESS=127.0.0.1`.

`./logs` steht im Container schreibgeschützt unter `/app/logs` bereit. Eigene
Optuna-Datenbanken und weitere Dateien nach `./data` legen; sie sind in DeepCAVE
unter `/app/data` erreichbar. Die Anwendung speichert ihren Cache und die
Auswahl der Runs im Docker-Volume `deepcave-cache`. `docker compose down -v`
entfernt dieses Volume und damit die gespeicherte Auswahl.

DeepCAVE bietet hier keine Anmeldung und der direkte Port verwendet HTTP. Soll
der Dienst über das Internet erreichbar sein, einen Reverse Proxy mit HTTPS und
Zugriffsschutz davor schalten und den DeepCAVE-Port nur auf `127.0.0.1` binden.
Bei direkter Freigabe muss TCP-Port 8050 (oder der gewählte Hostport) in der
Host- beziehungsweise Cloud-Firewall erreichbar sein.

### Lokal ohne DeepCAVE-Container

```bash
docker compose up -d redis
uv run deepcave
```
