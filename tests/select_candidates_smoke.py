"""Exercise the production selection callbacks in Chromium with two synthetic Optuna studies.

Run with: uv run --locked python -m tests.select_candidates_smoke
"""

import json
import logging
import pickle
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from urllib.parse import unquote

import optuna
from dash import Dash, dcc, html
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.trial import create_trial
from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server

import deepcave
from deepcave.plugins.select_candidates import SelectCandidates
from deepcave.runs.converters.optuna import OptunaRun

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def main():
    """Check bidirectional selection, persistence, hover, axes and the TOML download."""
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    with tempfile.TemporaryDirectory(prefix="candidate-smoke-") as directory:
        runs = []
        for index in range(2):
            study = optuna.create_study(study_name=f"study-{index}", direction="maximize")
            distributions = {
                "rate": FloatDistribution(0.001, 1, log=True),
                "model": CategoricalDistribution(["a", "b"]),
            }
            if index:
                distributions["extra"] = FloatDistribution(0, 1)
            for number in range(4):
                params = {"rate": 10 ** (-3 + number), "model": "a" if number % 2 else "b"}
                if index:
                    params["extra"] = 0.5
                trial = create_trial(
                    params=params,
                    distributions=distributions,
                    value=number + index / 2,
                    user_attrs={"score": 10 - number},
                )
                trial.datetime_start = datetime(2025, 1, 1)
                trial.datetime_complete = trial.datetime_start + timedelta(seconds=number + 1)
                study.add_trial(trial)
            path = Path(directory) / str(index)
            path.mkdir()
            with (path / "study.pkl").open("wb") as file:
                pickle.dump(study, file)
            runs.append(OptunaRun.from_path(path))
        app = Dash(
            __name__,
            assets_folder=str(deepcave.ROOT_DIR / "assets"),
            suppress_callback_exceptions=True,
        )
        cache = {}
        deepcave._api_mode = False
        deepcave.app = app
        deepcave.run_handler = SimpleNamespace(get_runs=lambda: runs)
        deepcave.rc = SimpleNamespace(
            get=lambda run, plugin, key: cache.get((run.id, plugin, key)),
            set=lambda run, plugin, key, value: cache.update({(run.id, plugin, key): value}),
        )
        plugin = SelectCandidates()
        app.layout = html.Div(
            [
                dcc.Dropdown(
                    id="global-run-selector",
                    options=[{"label": run.name, "value": run.id} for run in runs],
                    value=runs[0].id,
                ),
                dcc.Store(
                    id="global-main-objective-state",
                    data={"value": "objective:0", "direction": "upper"},
                ),
                *plugin(),
            ],
            style={"padding": "20px"},
        )
        server = make_server("127.0.0.1", 0, app.server, threaded=True)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
                page = browser.new_page(viewport={"width": 1600, "height": 1200})
                page.set_default_timeout(8000)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}")
                page.wait_for_function(
                    "document.querySelectorAll('#select-candidates-table td[data-dash-column=trial_number]').length === 4"
                )
                page.wait_for_function(
                    "document.querySelector('#select-candidates-scatter .js-plotly-plot')?.data?.[0]?.x?.length === 4"
                )
                page.evaluate(
                    "() => { window.candidateEvents = []; document.querySelector('#select-candidates-scatter .js-plotly-plot').on('plotly_selected', e => window.candidateEvents.push(e?.points?.map(p => p.customdata?.[2]))); }"
                )

                def count(number):
                    try:
                        page.wait_for_function(
                            "n => document.querySelector('#select-candidates-count').textContent === 'Selected Trials · ' + n",
                            arg=number,
                            timeout=8000,
                        )
                    except Exception:
                        print(
                            "Expected",
                            number,
                            "found",
                            page.locator("#select-candidates-count").inner_text(),
                            "errors",
                            errors,
                        )
                        print(
                            page.evaluate(
                                "() => ({events: window.candidateEvents, mode: document.querySelector('#select-candidates-scatter .js-plotly-plot')._fullLayout.dragmode})"
                            )
                        )
                        raise

                def plot_event(name, indices):
                    page.evaluate(
                        """args => {
                        const plot = document.querySelector('#select-candidates-scatter .js-plotly-plot');
                        plot.emit(args.name, {points: args.indices.map(index => ({customdata: plot.data[0].customdata[index]}))});
                    }""",
                        {"name": name, "indices": indices},
                    )

                def scatter_position(x, y):
                    return page.evaluate(
                        """values => {
                        const plot = document.querySelector('#select-candidates-scatter .js-plotly-plot');
                        const rect = plot.getBoundingClientRect(), layout = plot._fullLayout;
                        return {x: rect.x + layout.xaxis._offset + layout.xaxis.l2p(values[0]),
                                y: rect.y + layout.yaxis._offset + layout.yaxis.l2p(values[1])};
                    }""",
                        [x, y],
                    )

                first = page.locator(
                    '#select-candidates-table td[data-dash-column="trial_number"]'
                ).first
                first.click()
                count(1)
                first.click()
                count(0)
                first.locator("xpath=..").locator('input[type="checkbox"]').check()
                count(1)
                point = scatter_position(2, 1)
                page.mouse.click(point["x"], point["y"])
                count(2)
                page.mouse.move(0, 0)
                page.wait_for_function(
                    "document.querySelector('#select-candidates-scatter .js-plotly-plot').data[0].marker.size[1] === 9"
                )
                plot_event("plotly_click", [1])
                count(1)
                start, end = scatter_position(0.9, 2.1), scatter_position(3.1, -0.1)
                page.mouse.move(start["x"], start["y"])
                page.mouse.down()
                page.mouse.move(end["x"], end["y"], steps=12)
                page.mouse.up()
                count(3)
                # Use the actual lasso mode for a second additive group selection.
                page.locator(
                    '#select-candidates-scatter .modebar-btn[data-attr="dragmode"][data-val="lasso"]'
                ).click(force=True)
                page.wait_for_function(
                    "document.querySelector('#select-candidates-scatter .js-plotly-plot')._fullLayout.dragmode === 'lasso'"
                )
                corners = [
                    scatter_position(x, y)
                    for x, y in [(1.9, 0.9), (4.1, 0.9), (4.1, 3.1), (1.9, 3.1), (1.9, 0.9)]
                ]
                page.mouse.move(corners[0]["x"], corners[0]["y"])
                page.mouse.down()
                for corner in corners[1:]:
                    page.mouse.move(corner["x"], corner["y"], steps=8)
                page.mouse.up()
                count(4)
                page.mouse.move(0, 0)
                page.wait_for_function(
                    "document.querySelector('#select-candidates-parallel .js-plotly-plot').data?.[0]?.dimensions?.[0]?.values.length === 4"
                )

                # Table hover paints the corresponding scatter point and selected parallel line.
                first.hover()
                page.wait_for_function(
                    "document.querySelector('#select-candidates-scatter .js-plotly-plot').data[0].marker.size[0] === 13"
                )
                page.wait_for_function(
                    "document.querySelector('#select-candidates-parallel .js-plotly-plot').data[1].dimensions[0].values.length === 1"
                )
                page.mouse.move(0, 0)

                # Hit-test a real parallel line, then check the reverse hover direction.
                page.locator("#select-candidates-parallel").scroll_into_view_if_needed()
                geometry = page.evaluate("""() => {
                    const plot = document.querySelector('#select-candidates-parallel .js-plotly-plot');
                    const dimensions = plot.data[0].dimensions;
                    const axes = Array.from(plot.querySelectorAll('.parcoords .y-axis')).slice(0, dimensions.length);
                    if (axes.length !== dimensions.length) return {missing: true, axes: axes.length,
                        classes: [...new Set(Array.from(plot.querySelectorAll('g')).map(n => n.getAttribute('class')).filter(n => /axis|dimension|parcoord/.test(n)))]};
                    const rects = axes.map(axis => axis.querySelector('.axis .domain')?.getBoundingClientRect());
                    if (rects.some(rect => !rect)) return {axes: axes.length, missing: true};
                    const fraction = 0.3, row = 0;
                    const ys = rects.map((rect, i) => rect.bottom -
                        (dimensions[i].values[row] - dimensions[i].range[0]) /
                        (dimensions[i].range[1] - dimensions[i].range[0]) * rect.height);
                    return {x: rects[0].x * (1-fraction) + rects[1].x * fraction,
                            y: ys[0] * (1-fraction) + ys[1] * fraction};
                }""")
                assert not geometry.get("missing"), geometry
                page.mouse.move(geometry["x"], geometry["y"])
                page.wait_for_function(
                    "document.querySelector('.candidate-parallel-tooltip')?.textContent === 'study-0 · Trial 0'"
                )
                page.mouse.move(0, 0)

                page.evaluate(
                    "window.dash_clientside.set_props('select-candidates-axes', {value: ['rate']})"
                )
                page.wait_for_function(
                    "document.querySelector('#select-candidates-parallel .js-plotly-plot').data[0].dimensions.length === 3"
                )
                page.locator("#select-candidates-pareto").check()
                page.wait_for_function(
                    "document.querySelector('#select-candidates-scatter .js-plotly-plot').data.length === 2"
                )

                # Filtering and central objective changes preserve the independent selection.
                page.locator(
                    '#select-candidates-table input[placeholder="filter data..."]'
                ).first.fill("unmatched-study")
                page.locator(
                    '#select-candidates-table input[placeholder="filter data..."]'
                ).first.press("Enter")
                page.wait_for_function(
                    "document.querySelectorAll('#select-candidates-table td[data-dash-column=trial_number]').length === 0"
                )
                count(4)
                page.evaluate(
                    "window.dash_clientside.set_props('global-main-objective-state', {data: {value: 'user_attr:score', direction: 'upper'}})"
                )
                page.evaluate(
                    "window.dash_clientside.set_props('select-candidates-objectives', {value: []})"
                )
                page.wait_for_function(
                    "document.querySelectorAll('#select-candidates-table th[data-dash-column=\"metric:Objective0\"]').length === 0"
                )
                count(4)
                page.locator(
                    '#select-candidates-table input[placeholder="filter data..."]'
                ).first.fill("")
                page.locator(
                    '#select-candidates-table input[placeholder="filter data..."]'
                ).first.press("Enter")
                page.wait_for_function(
                    "document.querySelector('#select-candidates-scatter .js-plotly-plot').data[0]?.y[0] === 10"
                )

                page.evaluate(
                    "id => window.dash_clientside.set_props('select-candidates-studies', {value: [id]})",
                    runs[1].id,
                )
                page.wait_for_function(
                    "document.querySelectorAll('#select-candidates-table td[data-dash-column=trial_number]').length === 8"
                )
                count(4)
                page.evaluate(
                    "window.dash_clientside.set_props('select-candidates-studies', {value: []})"
                )
                page.wait_for_function(
                    "document.querySelectorAll('#select-candidates-table td[data-dash-column=trial_number]').length === 4"
                )

                # Native study switching retains candidates and allows adding another study.
                page.evaluate(
                    "id => window.dash_clientside.set_props('global-run-selector', {value: id})",
                    runs[1].id,
                )
                page.wait_for_function(
                    "document.querySelector('#select-candidates-status').textContent.startsWith('4 trials') && document.querySelector('#select-candidates-table td[data-dash-column=study_name]')?.textContent === 'study-1'"
                )
                count(4)
                plot_event("plotly_click", [0])
                count(5)
                page.wait_for_function(
                    "document.querySelector('#select-candidates-parallel-note').textContent.includes('Missing')"
                )
                href = page.locator("#select-candidates-export").get_attribute("href")
                exported = tomllib.loads(unquote(href.split(",", 1)[1]))
                assert {row["study_name"] for row in exported["selected_trials"]} == {
                    "study-0",
                    "study-1",
                }
                assert len(exported["selected_trials"]) == 5
                assert (
                    page.evaluate(
                        "() => new Set(Object.values(JSON.parse(sessionStorage.getItem('select-candidates-selection'))).map(row => row.color)).size"
                    )
                    == 5
                )
                page.screenshot(path="/tmp/deepcave-select-candidates.png", full_page=True)
                page.locator(".candidate-chip button").nth(1).click()
                count(4)

                # Session state is restored after a full reload as well.
                page.reload()
                count(4)
                page.locator("#select-candidates-clear").click()
                count(0)
                assert "disabled" in page.locator("#select-candidates-export").get_attribute(
                    "class"
                )
                assert not errors, errors
                browser.close()
                print(
                    json.dumps(
                        {
                            "result": "passed",
                            "checks": "selection, filters, studies, objective, hover, TOML, reload, clear",
                        }
                    )
                )
        finally:
            server.shutdown()
            thread.join()


if __name__ == "__main__":
    main()
