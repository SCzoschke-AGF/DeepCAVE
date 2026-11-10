"""Exercise linked sensitivity and candidate states in the production Dash callbacks.

Run with: uv run --locked python -m tests.sensitivity_smoke
"""

import json
import logging
import os
import pickle
import tempfile
from itertools import product
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import optuna
from dash import Dash, Input, Output, dcc, html
from optuna.distributions import FloatDistribution
from optuna.trial import create_trial
from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server

import deepcave
from deepcave.plugins.select_candidates import SelectCandidates
from deepcave.plugins.sensitivity_analysis import SensitivityAnalysis
from deepcave.runs.converters.optuna import OptunaRun
from deepcave.runs.handler import (
    DATA_SEED_ATTRIBUTE_CACHE_KEY,
    MODEL_SEED_ATTRIBUTE_CACHE_KEY,
)
from deepcave.utils.cache import Cache
from deepcave.utils.sensitivity_references import REFERENCE_CACHE_KEY


def main():
    """Check real browser selection, filters, tabs and the shared candidate collection."""
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    with tempfile.TemporaryDirectory(prefix="sensitivity-smoke-") as directory:
        study = optuna.create_study(study_name="seed-design", direction="maximize")
        study.set_user_attr("dataset_sha256", "seed-design-data")
        for x, m, d, repeat in product((0.25, 0.5), range(4), range(4), range(2)):
            study.add_trial(
                create_trial(
                    params={"rate": x},
                    distributions={"rate": FloatDistribution(0.001, 1, log=True)},
                    value=x + m + d + 0.1 * repeat,
                    user_attrs={
                        "model_rng": m,
                        "data_rng": d,
                        "fold_auc": [x + m, x + d],
                        "cv_split_ids": [f"{d}-a", f"{d}-b"],
                        "reference_score": x + m + d + 20,
                    },
                )
            )
        with (Path(directory) / "study.pkl").open("wb") as file:
            pickle.dump(study, file)
        run = OptunaRun.from_path(directory)
        database = Path(directory) / "references.db"
        original = optuna.create_study(
            study_name="original-hpo", storage=f"sqlite:///{database}", direction="maximize"
        )
        original.set_user_attr("dataset_sha256", "seed-design-data")
        for value in (0.25, 0.5):
            original.add_trial(
                create_trial(
                    params={"rate": value},
                    distributions={"rate": FloatDistribution(0.001, 1, log=True)},
                    value=value,
                    user_attrs={"model_rng": 42, "data_rng": 7, "reference_score": value + 20},
                )
            )
        app = Dash(
            __name__,
            assets_folder=str(deepcave.ROOT_DIR / "assets"),
            suppress_callback_exceptions=True,
        )
        cache = Cache(Path(directory) / "settings.json")
        cache.set("optuna_database", value=str(database))
        cache.set(MODEL_SEED_ATTRIBUTE_CACHE_KEY, value="trial:model_rng")
        cache.set(DATA_SEED_ATTRIBUTE_CACHE_KEY, value="trial:data_rng")
        prepared = {}
        deepcave._api_mode = False
        deepcave.app = app
        deepcave.c = cache
        deepcave.run_handler = SimpleNamespace(
            get_runs=lambda **_: [run], get_working_directory=lambda: Path(directory)
        )
        deepcave.rc = SimpleNamespace(
            get=lambda source, plugin, key: prepared.get((source.id, plugin, key)),
            set=lambda source, plugin, key, value: prepared.update(
                {(source.id, plugin, key): value}
            ),
        )
        sensitivity, candidates = SensitivityAnalysis(), SelectCandidates()
        app.layout = html.Div(
            [
                dcc.Location(id="route"),
                dcc.Dropdown(
                    id="global-run-selector",
                    options=[{"label": run.name, "value": run.id}],
                    value=run.id,
                ),
                dcc.Store(
                    id="global-main-objective-state",
                    data={"value": "objective:0", "direction": "upper"},
                ),
                html.Div(id="page"),
            ],
            style={"padding": "20px"},
        )

        @app.callback(Output("page", "children"), Input("route", "pathname"))
        def route(pathname):
            return candidates() if pathname == "/plugins/select_candidates" else sensitivity()

        server = make_server("127.0.0.1", 0, app.server, threaded=True)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=True,
                    args=["--no-sandbox"],
                    executable_path=os.environ.get("BROWSER_PATH"),
                )
                page = browser.new_page(viewport={"width": 1600, "height": 1200})
                page.set_default_timeout(10000)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}/plugins/sensitivity_analysis")
                assert page.locator("#sensitivity-mode").count() == 0
                page.wait_for_function(
                    "document.querySelector('#sensitivity-scatter .js-plotly-plot')"
                    "?.data?.[0]?.x?.length === 2"
                )

                def event(name, indices):
                    page.evaluate(
                        """args => {
                        const plot = document.querySelector('#sensitivity-scatter .js-plotly-plot');
                        plot.emit(args.name, {points: args.indices.map(i => ({
                            customdata: plot.data[0].customdata[i],
                            curveNumber: 0, pointNumber: i
                        }))});
                    }""",
                        {"name": name, "indices": indices},
                    )

                def activate_boxplot():
                    page.evaluate(
                        """() => {
                        const plot = document.querySelector('#sensitivity-distribution .js-plotly-plot');
                        const trace = plot.data.find(trace => trace.customdata?.length);
                        plot.emit('plotly_click', {points: [{customdata: trace.customdata[0]}]});
                    }"""
                    )

                target_label = page.evaluate(
                    "document.querySelector('#sensitivity-scatter .js-plotly-plot')"
                    ".data[0].hovertext[0].split('</b>')[0].slice(3)"
                )
                page.wait_for_function(
                    "document.querySelector('#sensitivity-trials-by-seed .js-plotly-plot')"
                    "?.data?.[0]?.y.length === 64"
                )
                page.locator("#sensitivity-trials-by-seed").screenshot(
                    path="/tmp/deepcave-trials-by-seed.png"
                )
                event("plotly_click", [0])
                page.wait_for_timeout(200)
                assert page.locator("#sensitivity-active-label").inner_text() != target_label
                activate_boxplot()
                page.wait_for_function(
                    "label => document.querySelector('#sensitivity-active-label')"
                    "?.textContent === label",
                    arg=target_label,
                )
                assert not page.locator("#sensitivity-table").is_visible()
                assert page.locator("#sensitivity-distribution").count() == 1
                assert page.locator("#sensitivity-distribution").is_visible()
                assert not page.locator("#sensitivity-reference-panel").is_visible()
                page.locator("#sensitivity-tab .tab").filter(has_text="Reference Trials").click()
                page.locator("#sensitivity-reference-panel").wait_for(state="visible")
                assert page.locator("#sensitivity-reference-controls .Select-control").count() == 2
                assert page.locator("#sensitivity-reference-search-status").inner_text() == ""
                assert cache.get(REFERENCE_CACHE_KEY) is None
                page.locator("#sensitivity-reference-search").click()
                page.wait_for_function(
                    "document.querySelector('#sensitivity-reference-search-status')"
                    "?.textContent.includes('Found 2 matching trials')"
                )
                page.locator("#sensitivity-reference-controls .Select-control").first.click()
                page.locator(
                    "#sensitivity-reference-controls .VirtualizedSelectOption"
                ).first.click()
                page.wait_for_function(
                    "document.querySelector('#sensitivity-reference-save-status')"
                    "?.textContent.includes('Reference trial saved')"
                )
                assert len(Cache(Path(directory) / "settings.json").get(REFERENCE_CACHE_KEY)) == 1
                for key, trial in cache.get(REFERENCE_CACHE_KEY).items():
                    legacy = {
                        name: value
                        for name, value in trial.items()
                        if name not in ("metrics", "user_attrs")
                    }
                    cache.set(REFERENCE_CACHE_KEY, key, value=legacy)
                page.reload()
                page.wait_for_function(
                    "document.querySelector('#sensitivity-scatter .js-plotly-plot')"
                    "?.data?.[0]?.x?.length === 2"
                )
                page.locator("#sensitivity-tab .tab").filter(has_text="Reference Trials").click()
                page.locator("#sensitivity-reference-panel").wait_for(state="visible")
                page.wait_for_function(
                    "document.querySelector('#sensitivity-reference-controls .Select-value-label')"
                    "?.textContent.includes('original-hpo')"
                )
                assert page.locator("#sensitivity-reference-search-status").inner_text() == ""
                page.locator("#sensitivity-tab .tab").filter(has_text="Distribution").click()
                activate_boxplot()
                event("plotly_selected", [0, 1])
                page.wait_for_function(
                    "document.querySelector('#sensitivity-scatter .js-plotly-plot')"
                    "?.data?.[0]?.marker.line.width.every(width => width >= 2)"
                )

                def tab(label, expected):
                    page.locator("#sensitivity-tab .tab").filter(has_text=label).click()
                    page.wait_for_function(
                        "expected => document.querySelector('#sensitivity-content')"
                        "?.textContent.includes(expected)",
                        arg=expected,
                    )

                for removed_tab in (
                    "Ranking Stability",
                    "Variance Decomposition",
                    "Seed Profiles",
                    "Mean Objective by Seed",
                ):
                    assert page.locator("#sensitivity-tab").get_by_text(removed_tab).count() == 0
                tab("Seed Effects", "Model seed")
                tab("Parameter sensitivity", "Descriptive association")
                tab("Distribution", "Trials by Seed per Configuration")
                page.evaluate(
                    "window.dash_clientside.set_props('global-main-objective-state', "
                    "{data: {value: 'user_attr:reference_score', direction: 'upper'}})"
                )
                page.wait_for_function(
                    "document.querySelector('#sensitivity-distribution .js-plotly-plot')"
                    "?.data?.[0]?.y?.every(value => value >= 20)"
                )
                assert page.locator("#sensitivity-show-reference-objectives").is_visible()
                page.evaluate("""async () => {
                        window.referenceTogglePlots = ['distribution', 'trials-by-seed', 'randomness-range']
                            .map(id => document.querySelector('#sensitivity-' + id + ' .js-plotly-plot'));
                        await Plotly.relayout(window.referenceTogglePlots[0], {
                            'yaxis.range': [20, 28], 'yaxis.autorange': false
                        });
                    }""")
                page.locator("#sensitivity-show-reference-objectives").check()
                page.wait_for_function(
                    "document.querySelector('#sensitivity-distribution .js-plotly-plot')"
                    "?.data?.find(trace => trace.name === 'Reference trial objective')?.y.length === 1"
                )
                assert page.evaluate(
                    "window.referenceTogglePlots.every((plot, index) => plot === "
                    "document.querySelector('#sensitivity-' + ['distribution', 'trials-by-seed', 'randomness-range'][index] + ' .js-plotly-plot'))"
                )
                assert page.evaluate("window.referenceTogglePlots[0].layout.yaxis.range") == [
                    20,
                    28,
                ]
                expected_reference = next(iter(cache.get(REFERENCE_CACHE_KEY).values()))[
                    "user_attrs"
                ]["reference_score"]
                assert (
                    page.evaluate(
                        "document.querySelector('#sensitivity-distribution .js-plotly-plot')"
                        ".data.find(trace => trace.name === 'Reference trial objective').y[0]"
                    )
                    == expected_reference
                )
                page.locator("#sensitivity-show-reference-objectives").uncheck()
                page.wait_for_function(
                    "!document.querySelector('#sensitivity-distribution .js-plotly-plot')"
                    "?.data?.some(trace => trace.name === 'Reference trial objective')"
                )
                assert page.evaluate(
                    "window.referenceTogglePlots.every((plot, index) => plot === "
                    "document.querySelector('#sensitivity-' + ['distribution', 'trials-by-seed', 'randomness-range'][index] + ' .js-plotly-plot'))"
                )
                assert page.evaluate("window.referenceTogglePlots[0].layout.yaxis.range") == [
                    20,
                    28,
                ]
                page.evaluate(
                    "window.dash_clientside.set_props('global-main-objective-state', "
                    "{data: {value: 'objective:0', direction: 'upper'}})"
                )
                page.wait_for_function(
                    "document.querySelector('#sensitivity-distribution .js-plotly-plot')"
                    "?.data?.[0]?.y?.every(value => value < 10)"
                )
                page.wait_for_function(
                    "document.querySelector('#sensitivity-scatter .js-plotly-plot')"
                    "?.data?.[0]?.y?.every(value => value < 10)"
                )
                activate_boxplot()
                page.wait_for_function(
                    "label => document.querySelector('#sensitivity-active-label')?.textContent === label",
                    arg=target_label,
                )
                page.locator("#sensitivity-trials tbody input[type=checkbox]").nth(1).check()
                page.locator("#sensitivity-add-trials").click()
                page.wait_for_function(
                    "document.querySelector('#sensitivity-candidate-status')"
                    "?.textContent.includes('1 trials added')"
                )
                page.locator("#sensitivity-add-config").click()
                page.wait_for_function(
                    "document.querySelector('#sensitivity-candidate-status')"
                    "?.textContent.includes('32 trials added')"
                )
                page.screenshot(path="/tmp/deepcave-sensitivity.png", full_page=True)
                page.get_by_text("Open Select Candidates", exact=True).click()
                page.wait_for_function(
                    "document.querySelector('#select-candidates-count')?.textContent.includes('32')"
                )
                page.locator(".candidate-chip a").first.click()
                page.wait_for_function(
                    "document.querySelector('#sensitivity-content')"
                    "?.textContent.includes('Trials by Seed per Configuration')"
                )
                assert page.locator("#sensitivity-distribution").count() == 1
                assert page.locator("#sensitivity-distribution").is_visible()
                assert not errors, json.dumps(errors)
                browser.close()
                print(
                    "Sensitivity browser smoke passed: linked active/comparison states, "
                    "tabs, filters and shared candidate persistence."
                )
        finally:
            server.shutdown()
            thread.join(timeout=5)


if __name__ == "__main__":
    main()
