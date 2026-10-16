# Copyright 2021-2024 The DeepCAVE Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  noqa: D400
"""
# Static

This module provides a plugin class for a static plugin.

It provides an Enum used for the plugin state.

## Classes
    - PluginState: An Enum to define the state of the Plugin.
    - StaticPlugin: This class provides a static plugin object.
"""

from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import traceback
from enum import Enum
from functools import wraps
from time import monotonic
from uuid import uuid4

import dash_bootstrap_components as dbc
from dash import ctx, dcc, html
from dash.dash import no_update
from dash.dependencies import Input, Output, State
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate

from deepcave import interactive
from deepcave.plugins import Plugin
from deepcave.runs import AbstractRun
from deepcave.utils.hash import string_to_hash
from deepcave.utils.url import create_url


class PluginState(Enum):
    """An Enum to define the state of the Plugin."""

    UNSET = -1
    READY = 0
    NEEDS_PROCESSING = 1
    PROCESSING = 2
    FAILED = 3


def _should_refresh_outputs(
    inputs_changed: bool,
    refresh_required: bool,
    triggered_id: Optional[str],
    delivery_pending: bool = False,
) -> bool:
    """Return whether cached outputs must be sent to the current browser page."""
    return inputs_changed or refresh_required or triggered_id is None or delivery_pending


def _should_update_status_content(
    previous_state: Optional[PluginState], state: PluginState, triggered_id: Optional[str]
) -> bool:
    """Return whether the current browser page needs a button-state update."""
    return triggered_id is None or previous_state != state


def _process_button_disabled(state: PluginState) -> bool:
    """Return whether starting another calculation would be unsafe."""
    return state in (PluginState.UNSET, PluginState.PROCESSING)


def _result_delivery_complete(
    state: PluginState, expected_key: Optional[str], rendered_key: Optional[str]
) -> bool:
    """Return whether the current page has received the ready result."""
    return (
        state == PluginState.READY
        and expected_key is not None
        and rendered_key == expected_key
    )


def _append_render_key(
    processed_outputs: Any, output_count: int, render_key: str
) -> List[Any]:
    """Append a delivery key without unpacking a single list-like Dash component."""
    outputs = [processed_outputs] if output_count == 1 else list(processed_outputs)
    return [*outputs, render_key]


def _delivery_attempt_due(
    page_id: Optional[str],
    render_key: str,
    rendered_key: Optional[str],
    attempts: Dict[Tuple[str, str], float],
    now: float,
    retry_after: float = 10.0,
) -> bool:
    """Allow one result delivery per page and retry it after a bounded timeout."""
    if not page_id or rendered_key == render_key:
        return False
    last_attempt = attempts.get((page_id, render_key))
    return last_attempt is None or now - last_attempt >= retry_after


def _job_is_active(queue: Any, job_id: str, submitted_job_ids: Set[str]) -> bool:
    """Keep a submitted job active across gaps between the queue registries."""
    return (
        job_id in submitted_job_ids
        or queue.is_running(job_id)
        or queue.is_pending(job_id)
    )


def _processing_inputs_key(plugin: "StaticPlugin", inputs: Dict[str, Any]) -> str:
    """Hash only values consumed by ``process``, excluding transient UI metadata."""
    processing_inputs: Dict[str, Any] = {}
    used_ids = set()
    for input_id, attribute, is_filter, _type in plugin.inputs:
        if is_filter or input_id in used_ids:
            continue
        processing_inputs[input_id] = {
            attribute: inputs.get(input_id, {}).get(attribute)
        }
        used_ids.add(input_id)
    if "_global_main_objective" in inputs:
        processing_inputs["_global_main_objective"] = inputs["_global_main_objective"]
    if getattr(plugin, "processing_version", 0):
        processing_inputs["_processing_version"] = plugin.processing_version
    return string_to_hash(str(processing_inputs))


def _applied_inputs_key(plugin: "StaticPlugin", inputs: Dict[str, Any]) -> str:
    """Identify processing inputs together with the globally selected run."""
    return string_to_hash(
        str(
            {
                "run": inputs.get("run", {}).get("value"),
                "processing": _processing_inputs_key(plugin, inputs),
            }
        )
    )


def _process_button_was_triggered(
    n_clicks: Optional[int], triggered_id: Any, button_id: str
) -> bool:
    """Return whether this callback invocation came from the process button itself."""
    return n_clicks is not None and triggered_id == button_id


def _render_inputs_key(plugin: "StaticPlugin", inputs: Dict[str, Any]) -> str:
    """Track view filters for browser delivery while keeping processing cache keys unchanged."""
    filters: Dict[str, Any] = {}
    for input_id, attribute, is_filter, _ in plugin.inputs:
        if is_filter and input_id not in filters:
            filters[input_id] = inputs.get(input_id, {}).get(attribute)
    return string_to_hash(str({"applied": _applied_inputs_key(plugin, inputs), "filters": filters}))


def _manage_callback_blocking(plugin: "StaticPlugin") -> Callable:
    """Block interval callbacks only while the decorated callback is running."""

    def decorator(callback: Callable) -> Callable:
        @wraps(callback)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            plugin._blocked = True
            try:
                return callback(*args, **kwargs)
            finally:
                plugin._blocked = False

        return wrapped

    return decorator


def _process(
    process: Callable[[AbstractRun, Any], None], run: AbstractRun, inputs: Dict[str, Any]
) -> None:
    """
    Process the run with the input data if possible.

    Return raw data based on a run and input data.

    Parameters
    ----------
    process : Callable[[AbstractRun, Any], None]
        The process function.
    run : AbstractRun
        The run.
    inputs : Dict[str, Any]
        The inputs as a dictionary.

    Raises
    ------
    Exception
        If the process function fails.
    """
    try:
        return process(run, inputs)
    except Exception:
        traceback.print_exc()
        raise


class StaticPlugin(Plugin, ABC):
    """
    Provide a static plugin object.

    Calculation with queue. Made for time-consuming tasks.
    Register and handle callbacks.

    Properties
    ----------
    outputs : List[Tuple[str, str, bool]]
        The plugin specific outputs.
    inputs : List[Tuple[str, str, bool, Any]]
        The plugin specific inputs.
    id : str
        The plugin id.
    raw_outputs : dict[str, Any]
        The raw outputs of a run.
    logger : Logger
        The logger for the plugin.
    name : str
        The name of the plugin.
    process : Callable
        Return raw data based on a run and input data.
    button_caption : str
        The caption for the button.
    """

    def __init__(self) -> None:
        super().__init__()
        self._setup()

    def _setup(self) -> None:
        """Set up the plugin."""
        self._state = PluginState.UNSET  # Set in the main loop to track what's going on right now
        self._previous_state: Optional[PluginState] = None  # Used for updating status
        self._refresh_required = True
        self._blocked = False
        self._expected_render_key: Optional[str] = None
        self._expected_render_keys: Dict[str, str] = getattr(self, "_expected_render_keys", {})
        # ``__call__`` resets the page-local state whenever the route is mounted. Keep submitted
        # jobs across that reset so navigating away cannot make a running job look completed.
        self._submitted_job_ids: Set[str] = getattr(self, "_submitted_job_ids", set())
        self._render_attempts: Dict[Tuple[str, str], float] = getattr(
            self, "_render_attempts", {}
        )

    @interactive
    def register_callbacks(self) -> None:
        """Register different callbacks."""
        super().register_callbacks()
        self._callback_inputs_changed()
        self._callback_loop_update_status_label()
        self._callback_loop_trigger_main_loop()

    @interactive
    def _callback_inputs_changed(self) -> None:
        """Handle callback if the inputs changed."""
        from deepcave import app, c, queue, rc, run_handler

        # Plugin specific outputs
        outputs = []
        for id, attribute in self.outputs:
            outputs.append(Output(self.get_internal_output_id(id), attribute))
        outputs.append(Output(self.get_internal_id("rendered-inputs-key"), "data"))

        inputs = [
            Input(self.get_internal_id("update-button"), "n_clicks"),
            Input(self.get_internal_id("update-interval-output"), "data"),
            Input("global-run-selector", "value"),
        ]

        # Get other plugin specific inputs that might change
        for id, attribute, _, _ in self.inputs:
            inputs.append(Input(self.get_internal_input_id(id), attribute))
        if self.main_objective_input is not None:
            inputs.append(Input("global-main-objective-state", "data"))
        inputs.append(State(self.get_internal_id("rendered-inputs-key"), "data"))
        inputs.append(State(self.get_internal_id("page-instance-id"), "data"))

        # Register updates from inputs
        @app.callback(outputs, inputs)  # type: ignore
        @_manage_callback_blocking(self)
        def plugin_process(
            n_clicks: int, _: Any, global_run_id: str, *inputs_list: str
        ) -> Optional[Any]:
            """Register updates from inputs."""
            rendered_inputs_key = inputs_list[-2]
            page_id = inputs_list[-1]
            inputs_list = inputs_list[:-2]
            global_objective_state = None
            if self.main_objective_input is not None:
                global_objective_state = inputs_list[-1]
                inputs_list = inputs_list[:-1]
            # Map the list `inputs_list` to a dict s.t.
            # it's easier to access them.
            inputs = self._list_to_dict(list(inputs_list), input=True)
            inputs["run"] = {"value": global_run_id}
            runs = self.get_selected_runs(inputs)
            for index, run in enumerate(runs):
                self.apply_global_main_objective(
                    inputs,
                    run,
                    global_objective_state,
                    include_cache_signature=index == 0,
                )
            inputs_key = _processing_inputs_key(self, inputs)
            cleaned_inputs = self._clean_inputs(inputs)
            last_inputs = c.get("last_inputs", self.id)

            link = create_url(self.get_base_url(), inputs)
            self.check_runs_compatibility(runs)

            button_pressed = _process_button_was_triggered(
                n_clicks,
                ctx.triggered_id,
                self.get_internal_id("update-button"),
            )
            inputs_changed = last_inputs is None or _applied_inputs_key(
                self, inputs
            ) != _applied_inputs_key(self, last_inputs)

            if button_pressed:
                self.logger.debug("Button pressed.")
                c.set("last_inputs", self.id, value=inputs)
                self._expected_render_key = _render_inputs_key(self, inputs)
                if page_id:
                    self._expected_render_keys[page_id] = self._expected_render_key
                for run in runs:
                    job_id = self._get_job_id(run.id, inputs_key)

                    # Do not duplicate an active or not-yet-collected job. A cached result does
                    # not prevent a new job: clicking Process explicitly requests recomputation.
                    if queue.is_processed(job_id):
                        continue

                    job_meta = {
                        "display_name": self.name,
                        "run_name": run.name,
                        "run_id": run.id,
                        "plugin_id": self.id,
                        "inputs_key": inputs_key,
                        "link": link,
                    }

                    self.logger.debug(f"Enqueued {run.name} ({run.id}).")
                    queue.enqueue(
                        _process,
                        args=[self.process, run, cleaned_inputs],
                        job_id=job_id,
                        meta=job_meta,
                    )
                    self._submitted_job_ids.add(job_id)

                self._refresh_required = True
                self._state = PluginState.PROCESSING
                raise PreventUpdate

            # Controls can change without applying the analysis, and delayed callbacks can still
            # carry values from a page that has already been replaced. Neither may overwrite the
            # last explicitly applied result.
            if last_inputs is not None and inputs_changed:
                if self._state != PluginState.PROCESSING:
                    self._state = PluginState.NEEDS_PROCESSING
                raise PreventUpdate

            # Collect completed jobs before reading the cache so a recomputed result can be shown
            # immediately on this callback cycle.
            for job in queue.get_finished_jobs():
                job_id = job.id
                if job.meta.get("plugin_id") != self.id:
                    continue
                try:
                    job_run_outputs = job.result
                    job_meta = job.meta
                    job_inputs_key = job_meta["inputs_key"]
                    job_run_id = job_meta["run_id"]
                    job_plugin_id = job_meta["plugin_id"]

                    self.logger.debug(f"Job {job_id} for run_id {job_meta['run_id']}.")
                    run = run_handler.get_run(job_run_id)
                    rc.set(run, job_plugin_id, job_inputs_key, job_run_outputs)
                    self.logger.debug(f"Job {job_id} cached.")

                    queue.delete_job(job_id)
                    self._submitted_job_ids.discard(job_id)
                    self.logger.debug(f"Job {job_id} deleted.")
                except Exception as e:
                    self.logger.error(f"Job {job_id} failed with exception {e}.")
                    queue.delete_job(job_id)
                    self._submitted_job_ids.discard(job_id)
                    self.logger.debug(f"Job {job_id} deleted.")
                except KeyboardInterrupt:
                    self.logger.error(f"Job {job_id} got interrupted by KeyboardInterrupt.")
                    queue.delete_job(job_id)
                    self._submitted_job_ids.discard(job_id)
                    self.logger.debug(f"Job {job_id} deleted.")
                    raise

            queue_active = False
            queue_failed = False
            for run in runs:
                job_id = self._get_job_id(run.id, inputs_key)
                queue_active = queue_active or _job_is_active(
                    queue, job_id, self._submitted_job_ids
                )
                if queue.has_failed(job_id):
                    queue_failed = True
                    queue.delete_job(job_id)
                    self._submitted_job_ids.discard(job_id)

            if queue_active:
                self._state = PluginState.PROCESSING
                raise PreventUpdate
            if queue_failed:
                self._state = PluginState.FAILED
                raise PreventUpdate

            raw_outputs = {run.id: rc.get(run, self.id, inputs_key) for run in runs}
            raw_outputs_available = all(output is not None for output in raw_outputs.values())
            if not raw_outputs_available:
                self._state = PluginState.NEEDS_PROCESSING
                raise PreventUpdate

            self._state = PluginState.READY

            # Dash invokes this callback when a page is mounted. Runtime flags on the plugin
            # instance are shared across browser pages, so ``_refresh_required`` may already be
            # false after an earlier page rendered the same cached result. Always return that
            # result on the initial call; otherwise a reload has an empty output.
            render_key = _render_inputs_key(self, inputs)
            now = monotonic()
            delivery_due = _delivery_attempt_due(
                page_id,
                render_key,
                rendered_inputs_key,
                self._render_attempts,
                now,
            )
            if rendered_inputs_key == render_key and page_id:
                self._render_attempts.pop((page_id, render_key), None)
            filter_changed = ctx.triggered_id in {
                self.get_internal_input_id(input_id)
                for input_id, _, is_filter, _ in self.inputs
                if is_filter
            }
            if _should_refresh_outputs(
                inputs_changed or filter_changed,
                self._refresh_required,
                ctx.triggered_id,
                delivery_due,
            ):
                self.raw_outputs = raw_outputs
                try:
                    processed_outputs = self._process_raw_outputs(inputs, raw_outputs, runs)
                except Exception:
                    # A cached result can be incompatible with a newer renderer. Do not leave
                    # READY waiting for a browser acknowledgement that can never arrive.
                    self._state = PluginState.FAILED
                    self._expected_render_key = None
                    if page_id:
                        self._expected_render_keys.pop(page_id, None)
                    raise
                self._expected_render_key = render_key
                if page_id:
                    self._expected_render_keys[page_id] = render_key
                self._refresh_required = False
                if page_id:
                    self._render_attempts[(page_id, render_key)] = now
                    if len(self._render_attempts) > 128:
                        oldest = min(
                            self._render_attempts,
                            key=self._render_attempts.get,  # type: ignore[arg-type]
                        )
                        self._render_attempts.pop(oldest, None)
                return _append_render_key(processed_outputs, len(self.outputs), render_key)

            raise PreventUpdate

    @interactive
    def _callback_loop_trigger_main_loop(self) -> None:
        from deepcave import app

        output = Output(self.get_internal_id("update-interval-output"), "data")
        inputs = [
            Input(self.get_internal_id("update-interval"), "n_intervals"),
            State(self.get_internal_id("update-interval-output"), "data"),
            State(self.get_internal_id("rendered-inputs-key"), "data"),
            State(self.get_internal_id("page-instance-id"), "data"),
        ]

        # Interval should not always run the main callback the whole time
        # Especially not if it's blocked because PreventUpdate
        # prevent output updates from previous callback calls.
        @app.callback(output, inputs)  # type: ignore
        def plugin_check_blocked(
            _: Any, data: Any, rendered_inputs_key: Optional[str], page_id: Optional[str]
        ) -> Any:
            """
            Check if blocked.

            Raises
            ------
            PreventUpdate
                If '_blocked' is True.
            """
            if self._blocked:
                raise PreventUpdate

            # The server-side callback ends before Dash receives and applies its response.
            # A newer interval invocation can supersede that response even if it returns
            # PreventUpdate. Keep polling quiet until this page acknowledges delivery, or
            # the existing retry timeout expires.
            expected_render_key = self._expected_render_keys.get(
                page_id or "", self._expected_render_key
            )
            if (
                self._state == PluginState.READY
                and page_id
                and expected_render_key is not None
                and rendered_inputs_key != expected_render_key
                and not _delivery_attempt_due(
                    page_id,
                    expected_render_key,
                    rendered_inputs_key,
                    self._render_attempts,
                    monotonic(),
                )
            ):
                raise PreventUpdate

            # This will trigger the main loop
            return data + 1

    @interactive
    def _callback_loop_update_status_label(self) -> None:
        from deepcave import app, notification

        output = [
            Output(self.get_internal_id("update-button"), "children"),
            Output(self.get_internal_id("update-button"), "disabled"),
        ]
        inputs = [
            Input(self.get_internal_id("update-interval"), "n_intervals"),
            Input(self.get_internal_id("rendered-inputs-key"), "data"),
            State(self.get_internal_id("page-instance-id"), "data"),
        ]

        # Update status label
        # Register updates from inputs
        @app.callback(output, inputs)  # type: ignore
        def plugin_update_status(
            _: Any, rendered_inputs_key: Optional[str], page_id: Optional[str] = None
        ) -> Tuple[Any, bool]:
            """Update the status of the plugin."""
            button_text: Any = [html.Span(self.button_caption)]
            expected_render_key = self._expected_render_keys.get(
                page_id or "", self._expected_render_key
            )

            if self._state == PluginState.UNSET:
                return button_text, True

            # Important so the button is not updated every time (would result in an ugly spinner).
            # An initial callback still has to synchronize a newly mounted page with the shared
            # plugin state, even if another page has already observed the same state.
            update_status_content = _should_update_status_content(
                self._previous_state, self._state, ctx.triggered_id
            ) or ctx.triggered_id == self.get_internal_id("rendered-inputs-key")
            if _result_delivery_complete(
                self._state, expected_render_key, rendered_inputs_key
            ):
                # Keep sending the spinner-free READY content. The callback that first clears
                # the spinner can be superseded by a closely following interval callback after
                # the result itself has already reached the browser.
                update_status_content = True
            if not update_status_content:
                # ``_previous_state`` is shared across browser pages. Only suppress replacement
                # of the unchanged children; the disabled flag below must still be returned so
                # one page cannot consume a state transition on behalf of another page.
                button_text = no_update

            # This is a special case where the main loop goes into "needs processing"
            # although the result is already there. This is because the queue needs a second
            # to be updated.
            if (
                self._previous_state == PluginState.PROCESSING
                and self._state == PluginState.NEEDS_PROCESSING
            ):
                # However: The previous state has to be unset, so if the inputs are really changed
                # the visualizer will be updated.
                self._previous_state = PluginState.UNSET
                raise PreventUpdate

            if self._state == PluginState.FAILED and update_status_content:
                notification.update(
                    "The job failed. Check the logs or make sure the worker is still running. "
                    "Most of the times, a simple restart might help."
                )

            disabled = _process_button_disabled(self._state)
            if self._state == PluginState.READY:
                disabled = not _result_delivery_complete(
                    self._state, expected_render_key, rendered_inputs_key
                )
            if self._state == PluginState.PROCESSING:
                if update_status_content:
                    button_text += [dbc.Spinner(size="sm", spinner_class_name="ms-2")]

            # Update the previous state
            self._previous_state = self._state

            return button_text, disabled

    def _get_job_id(self, run_name: str, inputs_key: str) -> str:
        """
        Get the job id.

        Parameters
        ----------
        run_name : str
            The name of the run.
        inputs_key : str
            The inputs key.

        Returns
        -------
        str
            The job id.
        """
        return f"{run_name}-{inputs_key}"

    @interactive
    def __call__(self) -> List[Component]:  # type: ignore
        """
        Return the components for the plugin.

        Basically, all blocks and elements of the plugin are stacked-up here.

        Returns
        -------
        List[Component]
            Layout as list of components.
        """
        from deepcave import config

        self._setup()

        components = [
            dcc.Interval(id=self.get_internal_id("update-interval"), interval=config.REFRESH_RATE),
            dcc.Store(id=self.get_internal_id("update-interval-output"), data=0),
            dcc.Store(id=self.get_internal_id("rendered-inputs-key")),
            dcc.Store(id=self.get_internal_id("page-instance-id"), data=str(uuid4())),
        ]
        components += super().__call__(True)

        return components
