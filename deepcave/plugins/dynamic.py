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
# Dynamic

This module provides a plugin class for a dynamic plugin.

Register and handle callbacks.

## Classes
    - DynamicPlugin: This class provides a dynamic plugin object.
"""

from abc import ABC
from typing import Any, Dict, List, Tuple

from dash.dependencies import Input, Output, State
from dash.development.base_component import Component

from deepcave import interactive
from deepcave.plugins import Plugin


class DynamicPlugin(Plugin, ABC):
    """
    Provide a dynamic plugin object.

    Register and handle callbacks.

    Properties
    ----------
    outputs : List[Tuple[str, str]]
        The registered outputs.
    inputs : List[Tuple[str, str, bool, Any]]
        The registered inputs.
    id : str
        The identifier of the Plugin.
    logger : Logger
        The logger for debugging information.
    use_cache : bool
        Defines whether to use the cache.
    raw_outputs : Dict[str, Any]
        A dictionary of the raw outputs.
    output_view_inputs : Tuple[Tuple[str, str, str], ...]
        External inputs passed only to ``load_outputs`` and excluded from processing caches.
    output_view_states : Tuple[Tuple[str, str, str], ...]
        External state read on regular updates and passed only to ``load_outputs``.
    """

    use_cache = True
    # Read-only inputs which redraw ``load_outputs`` without invalidating processed data.
    # Entries are ``(cleaned_input_name, component_id, component_property)``.
    output_view_inputs: Tuple[Tuple[str, str, str], ...] = ()
    output_view_states: Tuple[Tuple[str, str, str], ...] = ()

    def __init__(self) -> None:
        super().__init__()

    @interactive
    def register_callbacks(self) -> None:
        """
        Register basic callbacks for the plugin.

        Following callbacks are registered:
        - If inputs changes, the changes are pasted back.
          This is in particular interest if input dependencies are used.
        - Raw data dialog to display raw data.
        - Callback to be redirected to the config if clicked on it.
        """
        super().register_callbacks()
        from deepcave import app, c, rc

        outputs = []
        for id, attribute in self.outputs:
            outputs.append(Output(self.get_internal_output_id(id), attribute))

        inputs = [
            Input(self.get_internal_id("update-button"), "n_clicks"),
            Input("global-run-selector", "value"),
        ]
        for id, attribute, _, _ in self.inputs:
            inputs.append(Input(self.get_internal_input_id(id), attribute))
        if self.main_objective_input is not None:
            inputs.append(Input("global-main-objective-state", "data"))
        for _, component_id, component_property in self.output_view_inputs:
            inputs.append(Input(component_id, component_property))
        states = [
            State(component_id, component_property)
            for _, component_id, component_property in self.output_view_states
        ]

        # Register updates from inputs
        @app.callback(outputs, inputs, states)  # type: ignore
        def plugin_output_update(_: Any, global_run_id: str, *inputs_list: str) -> Any:
            """
            Update the outputs.

            Parameters
            ----------
            *inputs_list
                Input values from user.

            Returns
            -------
            Any
                The raw outputs.
            """
            # Map the list `inputs_list` to a dict s.t.
            # it's easier to access them.
            view_inputs: Dict[str, Any] = {}
            if self.output_view_states:
                state_values = inputs_list[-len(self.output_view_states) :]
                inputs_list = inputs_list[: -len(self.output_view_states)]
                view_inputs.update(
                    {
                        name: value
                        for (name, _, _), value in zip(
                            self.output_view_states, state_values
                        )
                    }
                )
            if self.output_view_inputs:
                view_values = inputs_list[-len(self.output_view_inputs) :]
                inputs_list = inputs_list[: -len(self.output_view_inputs)]
                view_inputs.update(
                    {
                        name: value
                        for (name, _, _), value in zip(self.output_view_inputs, view_values)
                    }
                )

            global_objective_state = None
            if self.main_objective_input is not None:
                global_objective_state = inputs_list[-1]
                inputs_list = inputs_list[:-1]
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
            # The Optional[str] would need tobe changed to a str as return type annotation
            # of the function _dict_as_key.
            inputs_key = self._dict_as_key(inputs, remove_filters=True)
            cleaned_inputs = self._clean_inputs(inputs)
            self.check_runs_compatibility(runs)

            raw_outputs = {}
            for run in runs:
                run_outputs = rc.get(run, self.id, inputs_key) if self.use_cache else None
                if run_outputs is None:
                    self.logger.debug(f"Process {run.name}.")
                    run_outputs = self.process(run, cleaned_inputs)

                    # Cache it
                    if self.use_cache:
                        rc.set(run, self.id, inputs_key, value=run_outputs)
                else:
                    self.logger.debug(f"Found outputs from {run.name} in cache.")

                raw_outputs[run.id] = run_outputs

            # Save for modal
            self.raw_outputs = raw_outputs

            # Cache last inputs
            c.set("last_inputs", self.id, value=inputs)

            return self._process_raw_outputs(inputs, raw_outputs, runs, view_inputs)

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
        return super().__call__(False)
