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
from deepcave.plugins.select_candidates import SelectCandidates, table_data
from deepcave.runs.converters.optuna import OptunaRun
from deepcave.utils.candidates import candidate_data, central_objective, scatter_figure

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def check_local_explorer(page, runs):
    """Compare browser calculations against Python for ties, directions and missing values."""
    from copy import deepcopy

    from dash.dash_table.Format import Format, Scheme
    from plotly.utils import PlotlyJSONEncoder

    data = candidate_data(runs[0])
    data["rows"].extend(candidate_data(runs[1])["rows"])
    data["rows"][0]["parameters"]["optional"] = None
    data["rows"][1]["parameters"]["optional"] = True
    presentation = {
        "scatter": scatter_figure([], None).to_plotly_json(),
        "numberFormat": Format(precision=6, scheme=Scheme.decimal_or_exponent).to_plotly_json(),
    }
    expected_rows, expected_columns = table_data(data, list(data["objectives"]))
    expected_columns = json.loads(json.dumps(expected_columns, cls=PlotlyJSONEncoder))
    actual, columns = page.evaluate(
        "args => window.dash_clientside.candidateExplorer.table(...args)",
        [data, list(data["objectives"]), presentation],
    )
    assert actual["rows"] == expected_rows
    assert columns == expected_columns
    for direction, elites, visible, mixed in [
        ("upper", False, None, False),
        ("lower", False, None, False),
        ("upper", True, None, False),
        ("lower", True, None, False),
        ("upper", True, [], False),
        ("upper", True, [data["rows"][0]["id"]], False),
        ("upper", True, None, True),
    ]:
        sample = deepcopy(data)
        for i, row in enumerate(sample["rows"]):
            row["metric_info"] = deepcopy(row["metric_info"])
            row["duration"] = [0, 1, 1, 2, 0, None, -1, 3][i]
            row["metrics"]["metric:Objective0"] = [5, 3, 3, 3, 6, 7, 8, None][i]
            row["metric_info"]["metric:Objective0"]["optimize"] = (
                "lower" if mixed and i == 0 else direction
            )
        state = {"value": "objective:0", "direction": direction}
        objective = central_objective(sample, state)
        rows = [row for row in sample["rows"] if visible is None or row["id"] in visible]
        expected = json.loads(scatter_figure(
            rows, objective, True, show_elites=elites, elite_fraction=0.5
        ).to_json())
        actual = page.evaluate(
            "args => window.dash_clientside.candidateExplorer.scatter(...args)",
            [visible, sample, state, True, elites, 0.5, presentation],
        )
        assert actual["data"] == expected["data"]
        for axis in ("xaxis", "yaxis"):
            assert actual["layout"][axis] == expected["layout"][axis]
        assert actual["layout"].get("annotations", []) == expected["layout"].get("annotations", [])
        assert json.loads(actual["layout"]["uirevision"]) == json.loads(expected["layout"]["uirevision"])


def main(selectors_only=False):
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
                check_local_explorer(page, runs)
                objectives = page.locator(".candidate-objectives .candidate-selector")
                studies_menu = page.locator(".candidate-studies .candidate-selector")
                summary = objectives.locator("summary")
                page.wait_for_function(
                    "document.querySelector('#select-candidates-objectives-summary').textContent === '2 objectives selected'"
                )
                assert summary.bounding_box()["height"] == studies_menu.locator("summary").bounding_box()["height"] == 36
                summary.click()
                checks = objectives.locator('input[type="checkbox"]')
                assert checks.count() == 2
                checks.first.uncheck()
                assert objectives.get_attribute("open") is not None
                page.wait_for_function(
                    "document.querySelector('#select-candidates-objectives-summary').textContent === 'score'"
                )
                search = objectives.locator('input[type="search"]')
                search.fill("not-an-objective")
                assert objectives.get_by_text("No matching options", exact=True).is_visible()
                assert objectives.locator('.candidate-selector-options label:visible').count() == 0
                objectives.get_by_role("button", name="Select all", exact=True).click()
                page.wait_for_function(
                    "document.querySelector('#select-candidates-objectives-summary').textContent === '2 objectives selected'"
                )
                search.fill("score")
                assert objectives.locator('.candidate-selector-options label:visible').count() == 1
                objectives.get_by_role("button", name="Clear all", exact=True).click()
                page.wait_for_function(
                    "document.querySelector('#select-candidates-objectives-summary').textContent === 'Select objectives …'"
                )
                search.fill("")
                assert objectives.locator('input[type="checkbox"]:checked').count() == 0
                objectives.get_by_role("button", name="Select all", exact=True).click()
                page.wait_for_function(
                    "document.querySelectorAll('#select-candidates-objectives input:checked').length === 2"
                )
                search.focus()
                page.keyboard.press("Escape")
                assert objectives.get_attribute("open") is None
                assert summary.evaluate("el => el === document.activeElement")
                studies_menu.locator("summary").click()
                studies_menu.locator('input[type="checkbox"]').first.check()
                page.wait_for_function(
                    "document.querySelectorAll('#select-candidates-table td[data-dash-column=trial_number]').length === 8"
                )
                assert studies_menu.get_attribute("open") is not None
                studies_menu.get_by_role("button", name="Clear all", exact=True).click()
                page.wait_for_function(
                    "document.querySelectorAll('#select-candidates-table td[data-dash-column=trial_number]').length === 4"
                )
                page.locator("#select-candidates-count").click()
                assert studies_menu.get_attribute("open") is None
                if selectors_only:
                    assert not errors, errors
                    page.screenshot(path="/tmp/deepcave-candidate-selectors.png", full_page=True)
                    browser.close()
                    print(json.dumps({"result": "passed", "checks": "compact selectors, search, checkboxes, bulk actions, studies, keyboard, field heights"}))
                    return
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

                # Pin selections even when sorting and selecting from another page.
                page.evaluate(
                    "window.dash_clientside.set_props('select-candidates-table', {page_size: 2, page_current: 1})"
                )
                plot_event("plotly_click", [3])
                count(1)
                page.wait_for_function(
                    "document.querySelector('#select-candidates-table td[data-dash-column=trial_number]')?.textContent === '3'"
                )
                page.wait_for_function("""() => {
                    const cell = document.querySelector('#select-candidates-table td[data-dash-column=trial_number]');
                    const selected = Object.values(JSON.parse(sessionStorage.getItem('select-candidates-selection')))[0];
                    const probe = document.createElement('span');
                    probe.style.color = selected.color;
                    document.body.appendChild(probe);
                    const matches = getComputedStyle(cell).color === getComputedStyle(probe).color;
                    probe.remove();
                    return matches;
                }""")
                page.evaluate(
                    "window.dash_clientside.set_props('select-candidates-table', {sort_by: [{column_id: 'duration', direction: 'asc'}], page_size: 20})"
                )
                page.wait_for_function(
                    "Array.from(document.querySelectorAll('#select-candidates-table td[data-dash-column=trial_number]')).map(n => n.textContent).join(',') === '3,0,1,2'"
                )
                plot_event("plotly_click", [1])
                count(2)
                page.wait_for_function(
                    "Array.from(document.querySelectorAll('#select-candidates-table td[data-dash-column=trial_number]')).map(n => n.textContent).join(',') === '1,3,0,2'"
                )
                page.locator(
                    "#select-candidates-table td[data-dash-column=trial_number]"
                ).first.locator("xpath=..").locator("input[type=checkbox]").click()
                count(1)
                page.wait_for_function(
                    "Array.from(document.querySelectorAll('#select-candidates-table td[data-dash-column=trial_number]')).map(n => n.textContent).join(',') === '3,0,1,2'"
                )
                page.locator("#select-candidates-clear").click()
                count(0)
                page.wait_for_function(
                    "Array.from(document.querySelectorAll('#select-candidates-table td[data-dash-column=trial_number]')).map(n => n.textContent).join(',') === '0,1,2,3'"
                )
                page.evaluate(
                    "window.dash_clientside.set_props('select-candidates-table', {sort_by: []})"
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
                    "document.querySelector('#select-candidates-scatter .js-plotly-plot').data[0].marker.size[1] === 11"
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
                page.wait_for_load_state("networkidle")
                requests = []
                page.on("request", lambda request: requests.append(request.url)
                        if request.method == "POST" else None)
                page.locator("#select-candidates-pareto").check()
                page.wait_for_function(
                    "document.querySelector('#select-candidates-scatter .js-plotly-plot').data.length === 2"
                )
                assert not requests, requests  # Pareto toggles must stay entirely in the browser.
                page.locator("#select-candidates-elites").check()
                page.wait_for_function(
                    "document.querySelector('#select-candidates-scatter .js-plotly-plot').data[0].customdata.some(row => row[4])"
                )
                assert not requests, requests
                page.locator("#select-candidates-elites").uncheck()

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
                    "document.querySelector('#select-candidates-table td[data-dash-column=study_name]')?.textContent === 'study-1'"
                )
                count(4)
                page.wait_for_function(
                    "document.querySelector('#select-candidates-scatter .js-plotly-plot').data[0]?.customdata?.[0]?.[1] === 'study-1'"
                )
                plot_event("plotly_click", [0])
                count(5)
                page.wait_for_function(
                    "document.querySelector('#select-candidates-parallel-note').textContent.includes('Missing')"
                )
                href = page.locator("#select-candidates-export").get_attribute("href")
                exported = tomllib.loads(unquote(href.split(",", 1)[1]))
                assert all(set(params) == {"rate", "model"} for params in exported["trials"])
                assert len(exported["trials"]) == 5
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
