"""Read-only browser smoke/latency probe for the persistent analysis lifecycle.

For an isolated, reproducible run, start DeepCAVE in a second terminal::

    .venv/bin/python -m tests.latency_fixture
    PYTHONPATH=$PWD .venv/bin/python deepcave/server.py --config tests/latency_config.py

Then run::

    .venv/bin/python -m tests.browser_navigation_smoke http://127.0.0.1:8061

The browser probe covers General, 1-D Analyses, 2-D Analyses and HPO Planner. It intentionally
does not import, refresh, or clear a study while measuring. By default it fails above 100 ms warm
navigation p95, 150 ms visible search-space commit, or 1000 ms background autosave. Override the
limits with ``DEEPCAVE_MAX_WARM_P95_MS``, ``DEEPCAVE_MAX_COMMIT_MS`` and
``DEEPCAVE_MAX_AUTOSAVE_MS`` when testing unusually large or remote setups.
"""

from typing import Any, Dict, List

import json
import os
import statistics
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROUTES = (
    "/plugins/one_dimensional_analysis",
    "/plugins/two_dimensional_analysis",
    "/plugins/hpo_planning",
)
ALL_ROUTES = ("/", *ROUTES)


def _p95(values: List[float]) -> float:
    """Return an inclusive p95 suitable for the small smoke sample."""
    return statistics.quantiles(values, n=20, method="inclusive")[18]


def _navigate(page: Page, route: str) -> float:
    """Use the application's dcc.Link and wait until its visible route is stable."""
    started = time.perf_counter()
    page.locator(f'a[href="{route}"]').first.click()
    page.wait_for_function("route => window.location.pathname === route", arg=route)
    if route in ROUTES:
        page.wait_for_function(
            """
            plugin => Array.from(document.querySelectorAll('.persistent-analysis-page'))
                .some(element => {
                    try { return JSON.parse(element.id).plugin === plugin; }
                    catch (_error) { return false; }
                })
            """,
            arg=route.rsplit("/", 1)[-1],
        )
    # Two animation frames are enough for visibility/Plotly resize work without baking a large
    # arbitrary delay into the reported navigation latency.
    page.wait_for_timeout(34)
    return 1000 * (time.perf_counter() - started)


def _commit_hpo_bound(page: Page) -> float:
    """Measure one committed planner edit through its visible derived preview."""
    bound = page.locator(".planner-bound-input").first
    bound.wait_for()
    row = bound.locator("xpath=ancestor::*[contains(@class, 'planner-parameter-row')]")
    range_control = row.locator(".planner-range-control")
    current_min = float(range_control.get_attribute("data-current-min"))
    current_max = float(range_control.get_attribute("data-current-max"))
    logarithmic = range_control.get_attribute("data-log") == "true"
    current_value = float(bound.input_value())
    fraction = 0.25
    proposal = (
        current_min * (current_max / current_min) ** fraction
        if logarithmic
        else current_min + fraction * (current_max - current_min)
    )
    step_value = range_control.get_attribute("data-step") or ""
    if step_value:
        step = float(step_value)
        proposal = current_min + round((proposal - current_min) / step) * step
    if range_control.get_attribute("data-parameter-type") == "integer":
        proposal = float(round(proposal))
    if abs(proposal - current_value) <= max(1e-8, abs(current_value) * 1e-8):
        fraction = 0.75
        proposal = (
            current_min * (current_max / current_min) ** fraction
            if logarithmic
            else current_min + fraction * (current_max - current_min)
        )
        if step_value:
            proposal = current_min + round((proposal - current_min) / step) * step
        if range_control.get_attribute("data-parameter-type") == "integer":
            proposal = float(round(proposal))

    preview = page.locator("#hpo-planning-dict-preview")
    previous_preview = preview.inner_text()
    started = time.perf_counter()
    bound.fill(str(proposal))
    bound.press("Tab")
    page.wait_for_function(
        "([id, previous]) => document.getElementById(id).textContent !== previous",
        arg=["hpo-planning-dict-preview", previous_preview],
    )
    elapsed = 1000 * (time.perf_counter() - started)
    assert abs(float(bound.input_value()) - proposal) <= max(1e-8, abs(proposal) * 1e-8)
    return elapsed


def main() -> None:
    """Exercise first and warm visits while collecting browser-visible timing facts."""
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8050"
    switches = int(os.environ.get("DEEPCAVE_BROWSER_SWITCHES", "20"))
    errors: List[str] = []
    callback_requests: List[Dict[str, int]] = []

    with sync_playwright() as playwright:
        chrome = os.environ.get("BROWSER_PATH") or (
            Path.home() / ".local/share/choreographer/deps/chrome-linux64/chrome"
        )
        browser = playwright.chromium.launch(
            executable_path=str(chrome),
            args=["--no-sandbox"],
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "console",
            lambda message: errors.append(message.text) if message.type == "error" else None,
        )
        page.on(
            "request",
            lambda request: (
                callback_requests.append({"request": len(request.post_data or ""), "response": 0})
                if request.url.endswith("/_dash-update-component")
                else None
            ),
        )

        page.goto(base_url, wait_until="domcontentloaded")
        page.locator("#general-page-container").wait_for(state="visible")
        page.evaluate(
            "document.querySelector('#general-page-container').dataset.smokeIdentity = 'stable'"
        )

        cold: Dict[str, float] = {}
        for route in ROUTES:
            cold[route] = _navigate(page, route)

        autosave_started = time.perf_counter()
        with page.expect_response(
            lambda response: response.url.endswith("/_dash-update-component")
            and "planned-search-space-persistence-revision" in (response.request.post_data or ""),
            timeout=5000,
        ):
            commit = _commit_hpo_bound(page)
        autosave = 1000 * (time.perf_counter() - autosave_started)
        bound = page.locator(".planner-bound-input").first
        proposal = float(bound.input_value())

        # A full reload must hydrate the committed value from DeepCAVE's metacache.
        page.reload(wait_until="domcontentloaded")
        bound = page.locator(".planner-bound-input").first
        bound.wait_for()
        assert abs(float(bound.input_value()) - proposal) <= max(1e-8, abs(proposal) * 1e-8)
        page.evaluate(
            "document.querySelector('#general-page-container').dataset.smokeIdentity = 'stable'"
        )

        _navigate(page, ROUTES[1])
        _navigate(page, ROUTES[2])
        bound = page.locator(".planner-bound-input").first
        bound.wait_for()
        assert abs(float(bound.input_value()) - proposal) <= max(1e-8, abs(proposal) * 1e-8)

        # Reload clears retained pages. Warm every measured route once before collecting the
        # repeated-navigation distribution so the cold mount is not mixed into its p95.
        for route in ALL_ROUTES:
            _navigate(page, route)

        warm = []
        route_cycle = ALL_ROUTES
        for index in range(switches):
            warm.append(_navigate(page, route_cycle[index % len(route_cycle)]))

        _navigate(page, "/")
        assert page.locator("#general-page-container").get_attribute("data-smoke-identity") == (
            "stable"
        )
        assert page.locator(".persistent-analysis-page").count() <= len(ROUTES)
        assert not errors, errors
        browser.close()

    results: Dict[str, Any] = {
        "cold_route_ms": {key: round(value, 1) for key, value in cold.items()},
        "warm_route_ms": {
            "median": round(statistics.median(warm), 1),
            "p95": round(_p95(warm), 1),
        },
        "search_space_commit_ms": round(commit, 1),
        "search_space_autosave_ms": round(autosave, 1),
        "dash_callback_requests": len(callback_requests),
    }
    limits = {
        "warm_p95": float(os.environ.get("DEEPCAVE_MAX_WARM_P95_MS", "100")),
        "commit": float(os.environ.get("DEEPCAVE_MAX_COMMIT_MS", "150")),
        "autosave": float(os.environ.get("DEEPCAVE_MAX_AUTOSAVE_MS", "1000")),
    }
    results["limits_ms"] = limits
    print(json.dumps(results, indent=2, sort_keys=True))
    result_file = os.environ.get("DEEPCAVE_LATENCY_RESULT")
    if result_file:
        Path(result_file).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    violations = []
    if _p95(warm) > limits["warm_p95"]:
        violations.append("warm navigation p95")
    if commit > limits["commit"]:
        violations.append("visible search-space commit")
    if autosave > limits["autosave"]:
        violations.append("background search-space autosave")
    assert not violations, f"Latency limits exceeded: {', '.join(violations)}"


if __name__ == "__main__":
    main()
