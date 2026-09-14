"""Read-only browser smoke/latency probe for the persistent analysis lifecycle.

Start DeepCAVE normally, then run::

    .venv/bin/python -m tests.browser_navigation_smoke http://127.0.0.1:8050

The script intentionally does not import, refresh, or clear any study.
"""

from typing import Dict, List

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

        # Commit one original-value HPO bound and navigate immediately. Returning must hydrate the
        # retained row from canonical browser memory rather than from a page default.
        bound = page.locator(".planner-bound-input").first
        bound.wait_for()
        row = bound.locator("xpath=ancestor::*[contains(@class, 'planner-parameter-row')]")
        range_control = row.locator(".planner-range-control")
        current_min = float(range_control.get_attribute("data-current-min"))
        current_max = float(range_control.get_attribute("data-current-max"))
        logarithmic = range_control.get_attribute("data-log") == "true"
        proposal = (
            (current_min * current_max) ** 0.5
            if logarithmic
            else current_min + 0.25 * (current_max - current_min)
        )
        step_value = range_control.get_attribute("data-step") or ""
        if step_value:
            step = float(step_value)
            proposal = current_min + round((proposal - current_min) / step) * step
        if range_control.get_attribute("data-parameter-type") == "integer":
            proposal = float(round(proposal))
        bound.fill(str(proposal))
        bound.press("Tab")
        _navigate(page, ROUTES[1])
        _navigate(page, ROUTES[2])
        bound = page.locator(".planner-bound-input").first
        bound.wait_for()
        assert abs(float(bound.input_value()) - proposal) <= max(1e-8, abs(proposal) * 1e-8)

        warm = []
        pair = ROUTES[1:]
        for index in range(switches):
            warm.append(_navigate(page, pair[index % 2]))

        _navigate(page, "/")
        assert page.locator("#general-page-container").get_attribute("data-smoke-identity") == (
            "stable"
        )
        assert page.locator(".persistent-analysis-page").count() <= len(ROUTES)
        assert not errors, errors
        browser.close()

    print("Cold route times (ms):", {key: round(value, 1) for key, value in cold.items()})
    print(
        "Warm route time (ms):",
        {"median": round(statistics.median(warm), 1), "p95": round(_p95(warm), 1)},
    )
    print("Dash callback requests:", len(callback_requests))


if __name__ == "__main__":
    main()
