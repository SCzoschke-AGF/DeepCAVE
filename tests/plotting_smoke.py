"""Manual browser/export smoke test: python -m tests.plotting_smoke.

Requires Playwright and Kaleido's Chrome installation. Outputs go to cache/plotting-smoke.
"""

import os
import sys
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import numpy as np
import pandas as pd
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from dash import Input, Output, dcc, html
from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server

from deepcave import get_app
from deepcave.plugins.hyperparameter import one_dimensional_analysis as analysis
from deepcave.plugins.hyperparameter.importances import Importances
from deepcave.plugins.objective.cost_over_time import CostOverTime
from deepcave.utils.layout import get_slider_marks


def main():
    """Render actual plugin figures and exercise a Dash callback in Chrome."""
    target = Path("cache/plotting-smoke")
    target.mkdir(parents=True, exist_ok=True)
    objective = SimpleNamespace(name="Objective0", optimize="upper")
    hp = UniformFloatHyperparameter("learning_rate", lower=0.004, upper=0.3, log=True)
    rng = np.random.default_rng(12)
    x = np.geomspace(0.004, 0.3, 500)
    trials = pd.DataFrame(
        dict(
            trial_number=range(500),
            hyperparameter_value=x,
            objective=0.18 - 0.08 * x + rng.normal(0, 0.006, 500),
        )
    )
    runs = [
        SimpleNamespace(
            id=str(i),
            name=f"xgboost_data-4897daef3195_average-317607933c6e-study-{i}",
            history=[],
            configspace={"learning_rate": hp},
            get_objective=lambda _: objective,
            get_budget=lambda *args, **kwargs: "Full budget",
        )
        for i in range(2)
    ]
    outputs = {
        run.id: dict(
            ids=[0, 1, 2],
            times=[1, 2000, 50000],
            costs_mean=[0.15, 0.17, 0.18],
            costs_std=[0, 0, 0],
            objective_values=trials.objective.tolist(),
        )
        for run in runs
    }
    figures = {
        "cost": CostOverTime.load_outputs(
            runs,
            dict(study_ids=["0", "1"], objective_id=0, xaxis="times"),
            outputs,
        )
    }
    summary, tail, threshold, count = analysis._summarize_trials(trials, hp, "upper", 0.1)
    panels = analysis.OneDimensionalAnalysis.load_outputs(
        runs[0],
        dict(
            objective_id=0,
            hyperparameter_name="learning_rate",
            elite_proportion=0.1,
            process_elite=1,
        ),
        dict(
            trials=trials.to_dict("records"),
            summary=summary,
            tail_quantile=tail,
            elite_threshold=threshold,
            elite_count=count,
        ),
    )
    figures.update(zip(("trials", "performance", "enrichment"), panels))
    names = [
        "learning_rate",
        "n_estimators",
        "max_depth",
        "min_child_weight",
        "colsample_bytree",
        "reg_alpha",
        "subsample",
        "gamma",
        "reg_lambda",
    ]
    figures["importances"] = Importances.load_outputs(
        runs[0],
        dict(
            objective_id2=None, hyperparameter_names=names, budget_ids=[0], n_hps=9, method="local"
        ),
        {0: {name: (0.7 / (i + 1) ** 3, 0.001) for i, name in enumerate(names)}},
    )
    for name, figure in ([] if "--browser-only" in sys.argv else figures.items()):
        for extension in ("png", "pdf"):
            path = target / f"{name}.{extension}"
            figure.write_image(path, width=1200, height=figure.layout.height)
            assert path.stat().st_size > 1000

    app = get_app("Plotting smoke test")
    app.layout = html.Div(
        [
            dcc.Slider(
                id="slider",
                min=0,
                max=10,
                value=0,
                step=None,
                marks=get_slider_marks(list(range(11))),
            ),
            html.Div(id="slider-value"),
            *[
                dcc.Graph(id=name, figure=figure, style={"width": "100%"})
                for name, figure in figures.items()
            ],
        ]
    )

    @app.callback(Output("slider-value", "children"), Input("slider", "value"))
    def slider_value(value):
        return str(value)

    server = make_server("127.0.0.1", 0, app.server)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            chrome = os.environ.get("BROWSER_PATH") or (
                Path.home() / ".local/share/choreographer/deps/chrome-linux64/chrome"
            )
            browser = p.chromium.launch(executable_path=str(chrome), args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1280, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
            page.goto(f"http://127.0.0.1:{server.server_port}")
            page.wait_for_function("document.querySelectorAll('.js-plotly-plot').length === 5")
            page.screenshot(path=str(target / "browser-initial.png"), full_page=True)
            page.wait_for_function("document.querySelector('#slider-value').textContent === '0'")
            page.locator(".rc-slider-mark-text").nth(1).click()
            page.wait_for_function(
                "Number(document.querySelector('#slider-value').textContent) === 1"
            )
            for name in figures:
                page.locator(f"#{name} .main-svg").first.wait_for()
                page.locator(f"#{name}").screenshot(path=str(target / f"{name}-dash.png"))
            assert page.locator("#cost .scatterlayer .trace").count() > 0
            assert page.locator("#importances .barlayer .point").count() == len(names)
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        thread.join()
    print(f"PASS: 5 Dash figures and slider callback; artifacts in {target}")
    if "--browser-only" not in sys.argv:
        print("PASS: 5 PNG and 5 PDF exports")


if __name__ == "__main__":
    main()
