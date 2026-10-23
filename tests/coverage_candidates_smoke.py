"""Real-study browser smoke test: uv run --locked python -m tests.coverage_candidates_smoke.

Uses the production layout, backend and callbacks. The local queue replaces only
Redis/RQ transport; every displayed metric is calculated from the source study.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

from dash import Dash, dcc, html
from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server

from deepcave import ROOT_DIR
from deepcave.evaluators.coverage import prepare_coverage, run_coverage_analysis
from deepcave.evaluators.coverage_candidates import score_coverage_candidates
from deepcave.evaluators.coverage_planning import generate_coverage_trial_plan
from deepcave.plugins.hyperparameter.combinatorial_views import combinatorial_view
from deepcave.plugins.hyperparameter.coverage_planning_views import plan_view
from deepcave.plugins.hyperparameter.coverage_views import _id
from deepcave.plugins.hyperparameter.performance_coverage_views import (
    register_performance_callbacks,
)
from deepcave.plugins.hyperparameter.search_space_coverage import (
    CoverageJobs,
    SearchSpaceCoverage,
)
from deepcave.runs.converters.optuna import OptunaRun


def walk(component):
    """Visit mounted production components, including lists."""
    if isinstance(component, (list, tuple)):
        for child in component:
            yield from walk(child)
    elif hasattr(component, "children"):
        yield component
        yield from walk(component.children)
    else:
        yield component


class LocalQueue:
    """Execute a cached worker result locally and exercise normal queue polling."""

    def __init__(self):
        self.jobs = {}
        self.calls = 0
        self.submitted = Event()

    def ready(self):
        return True

    def enqueue(self, function, args, job_id, meta):
        self.calls += 1
        job = SimpleNamespace(result=function(*args))
        job.get_status = lambda refresh=True: "finished"
        self.jobs[job_id] = job
        self.submitted.set()

    def get_job(self, job_id):
        return self.jobs[job_id]

    def delete_job(self, job_id):
        self.jobs.pop(job_id)


def main():
    """Compute real data, render the dashboard and exercise its interactions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="logs/Optuna/nn_fashionmnist_mo/run_1")
    parser.add_argument("--objective", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("cache/coverage-candidates-demo"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run = OptunaRun.from_path(args.run)
    queue = LocalQueue()
    jobs = CoverageJobs(args.output, queue)
    result = score_coverage_candidates(
        run, args.objective, str(jobs.directory / "_coverage_candidates")
    )
    jobs._write(run.id, "candidate_scoring", {"status": "completed", "result": result})
    data = prepare_coverage(run)
    baseline = run_coverage_analysis(
        "combinatorial_baseline",
        data,
        run.configspace,
        {
            "designs": 2,
            "seed": 42,
            "forecast": {
                "probability": 0.8,
                "simulations": 32,
                "max_trials": 100000,
                "timeout": 2,
                "seed": 42,
            },
        },
    )
    covering_plan = generate_coverage_trial_plan(
        data,
        {
            "strategy": "minimize_trials",
            "resolution": 2,
            "strength_mode": "next",
            "solver_timeout": 2,
            "seed": 42,
        },
    )
    plugin = SearchSpaceCoverage()
    layout = SearchSpaceCoverage.__call__.__wrapped__(plugin)
    workspace = next(
        c for c in walk(layout) if getattr(c, "className", None) == "coverage-global-layout"
    )
    mounted = {getattr(c, "id", None): c for c in walk(workspace) if hasattr(c, "id")}
    mounted[_id("combinatorial-result")].children = combinatorial_view(
        baseline["observed"], baseline, baseline["forecast"]
    )
    mounted[_id("planning-result")].children = plan_view(covering_plan)
    mounted[_id("combinatorial_baseline-status")].children = ""
    app = Dash(
        __name__,
        assets_folder=str(ROOT_DIR / "assets"),
        suppress_callback_exceptions=True,
        title="Lighthouse · Coverage candidates",
    )
    app.layout = html.Main(
        [
            html.Div(
                [
                    html.Strong("Lighthouse"),
                    html.Span(
                        "Search-space coverage · Fashion-MNIST · "
                        f"{len(run.history)} observed trials · "
                        f"{run.get_objective(args.objective).name}"
                    ),
                ],
                className="coverage-demo-heading",
            ),
            dcc.Store(id=_id("context"), data={"run_id": run.id}),
            dcc.Store(
                id="global-main-objective-state", data={"value": f"objective:{args.objective}"}
            ),
            dcc.Dropdown(
                id="global-run-selector",
                value=run.id,
                options=[{"label": run.name, "value": run.id}],
                style={"display": "none"},
            ),
            dcc.Interval(id=_id("tick"), interval=400),
            workspace,
        ],
        style={"padding": "20px", "background": "#faf9fd", "fontFamily": "Arial, sans-serif"},
    )
    # Keep the screenshot independent of remote CSS availability.
    app.index_string = app.index_string.replace(
        "</head>",
        """<style>
        body { margin: 0; } * { box-sizing: border-box; }
        .coverage-demo-heading { display:flex; gap:20px; align-items:center; margin-bottom:18px;
          color:#292340; } .coverage-demo-heading span { color:#77718d; font-size:12px; }
        button { cursor:pointer; border:1px solid #d5cee2; background:white; color:#5d5571;
          border-radius:4px; padding:6px 12px; }
        .btn-primary { background:#7752bf; color:white; }
        h3,h4,h5 { margin-top:0; } label { display:block; }
        input { border:1px solid #ddd7e9; border-radius:4px; padding:5px; width:100%; }
        .coverage-card { background:white; }
    </style></head>""",
    )
    register_performance_callbacks(app, jobs, SimpleNamespace(get_run=lambda _: run))
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    server = make_server("127.0.0.1", 0, app.server)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            chrome = os.environ.get("BROWSER_PATH") or str(
                Path.home() / ".local/share/choreographer/deps/chrome-linux64/chrome"
            )
            browser = p.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1600, "height": 1100}, device_scale_factor=1)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}")
            page.locator(f"#{_id('performance-candidates')}").get_by_text(
                f"{len(result['candidates']):,}", exact=True
            ).wait_for()
            page.locator(f"#{_id('performance-details')}").get_by_text(
                "Original KDE score"
            ).wait_for()
            assert (
                page.locator(f"#{_id('performance-top')}").inner_text()
                == f"{result['top_relative_score']:+.3f}"
            )
            plot = f"#{_id('performance-chart')} .js-plotly-plot"
            assert page.locator(plot).evaluate("p => p.layout.shapes[0].x0") == 0
            page.screenshot(path=str(args.output / "dashboard.png"), full_page=True)

            page.get_by_text("Best candidates: full search space", exact=True).click()
            page.get_by_text("Full-space search", exact=True).wait_for()
            table = page.locator(f"#{_id('performance-ranking')}")
            assert "Coverage hole" not in table.inner_text()
            assert "Hole radius" not in table.inner_text()
            assert "Toggle Columns" not in table.inner_text()
            count = len(result["full_space_candidates"])
            assert page.locator(f"#{_id('performance-candidates')}").inner_text() == f"{count:,}"
            assert (
                f"{count:,} sampled, unseen configurations"
                in page.locator(f"#{_id('performance-pool-summary')}").inner_text()
            )
            page.wait_for_function(
                "([selector, count]) => document.querySelector(selector).data[0].y"
                ".reduce((a, b) => a + b, 0) === count",
                arg=[plot, count],
            )
            page.get_by_text("Show full config", exact=True).click()
            config = json.loads(page.locator(".coverage-candidate-full-config pre").inner_text())
            best_global = sorted(
                result["full_space_candidates"], key=lambda c: (-c["kde_score"], c["id"])
            )[0]
            assert config == best_global["configuration"]
            table.locator("tr").nth(1).locator('input[type="checkbox"]').check()
            page.locator(
                f"#{_id('performance-export')} a[download='full-space-candidates.toml']"
            ).wait_for()
            page.screenshot(path=str(args.output / "full-space.png"), full_page=True)
            page.get_by_text("Show full config", exact=True).click()

            page.get_by_text("Recommended trials", exact=True).click()
            page.get_by_text("Included in CHR plan · #1", exact=True).wait_for()
            table = page.locator(f"#{_id('performance-ranking')}")
            row = table.locator("tr").nth(1)
            assert row.evaluate("r => getComputedStyle(r).display") == "table-row"
            visible_columns = row.locator("td").evaluate_all(
                "cells => cells.map(c => ({id: c.dataset.dashColumn || 'checkbox', "
                "x: c.getBoundingClientRect().x})).sort((a, b) => a.x - b.x).map(c => c.id)"
            )
            assert visible_columns == [
                "rank",
                "relative_score",
                "hole_id",
                "hole_radius",
                "checkbox",
            ]
            assert row.evaluate(
                "r => { const bounds = r.closest('.dash-table-container').getBoundingClientRect(); "
                "return [...r.cells].every(c => { const b = c.getBoundingClientRect(); "
                "return b.width > 0 && b.left >= bounds.left - 1 && b.right <= bounds.right + 1; }); }"
            )
            row.locator('input[type="checkbox"]').check()
            page.locator(f"#{_id('performance-export')} a[download]").wait_for()
            table.locator('td[data-dash-row="1"][data-dash-column="relative_score"]').click()
            page.get_by_text("Included in CHR plan · #2", exact=True).wait_for()
            page.get_by_text("Show full config", exact=True).click()
            config = json.loads(page.locator(".coverage-candidate-full-config pre").inner_text())
            chosen = next(c for c in result["candidates"] if c["id"] == result["plan"][1])
            assert config == chosen["configuration"]
            page.screenshot(path=str(args.output / "selected-plan.png"), full_page=True)
            page.get_by_text("Show full config", exact=True).click()
            page.get_by_text("Best candidate per hole", exact=True).click()
            assert "KDE vs. reference" in table.inner_text()
            assert queue.calls == 0  # Tabs and row selection never fit or plan.
            page.locator(f"#{_id('performance-run')}").click()
            assert queue.submitted.wait(timeout=15)
            page.wait_for_function(
                "() => !document.querySelector('#search_space_coverage-performance-run').disabled"
            )
            assert queue.calls == 1
            assert not errors, errors
            page.get_by_text("Best overall candidates", exact=True).click()
            page.set_viewport_size({"width": 600, "height": 1000})
            page.screenshot(path=str(args.output / "mobile.png"), full_page=True)
            browser.close()
    finally:
        server.shutdown()
        thread.join()
    (args.output / "demo-metadata.json").write_text(
        json.dumps(
            {
                "source": args.run,
                "study_name": run.name,
                "historical_trials": len(run.history),
                "objective": run.get_objective(args.objective).name,
                "holes_analyzed": len(result["holes"]),
                "generated_candidates": len(result["candidates"]),
                "full_space_sampling": result["global_sampling"],
                "plan_size": len(result["plan"]),
                "reference_score": result["reference_score"],
                "top_relative_score": result["top_relative_score"],
                "reference_trial_number": result["reference_trial_number"],
                "reference_objective": result["reference_objective"],
            },
            indent=2,
        )
    )
    print(
        "PASS: real-study histogram, tabs, detail selection, full config and Refresh; "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
