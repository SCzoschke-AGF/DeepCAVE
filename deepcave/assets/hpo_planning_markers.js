(function () {
    "use strict";

    const DEFAULT_TOP_N = 5;
    const TOP_N_LIMIT = 100;
    const SLIDER_EXTENSION = 0.25;

    function topTrialCount(value) {
        if (value === null || value === undefined || value === "") {
            return DEFAULT_TOP_N;
        }
        const parsed = Math.trunc(Number(value));
        if (!Number.isFinite(parsed)) {
            return DEFAULT_TOP_N;
        }
        return Math.min(TOP_N_LIMIT, Math.max(1, parsed));
    }

    function triggeredId() {
        const context = window.dash_clientside.callback_context;
        if (context && context.triggered_id !== undefined) {
            return context.triggered_id;
        }
        const triggered = context && context.triggered && context.triggered[0];
        if (!triggered || !triggered.prop_id) {
            return null;
        }
        const rawId = triggered.prop_id.slice(0, triggered.prop_id.lastIndexOf("."));
        try {
            return JSON.parse(rawId);
        } catch (_error) {
            return rawId;
        }
    }

    function transformValue(value, logarithmic) {
        const numeric = Number(value);
        if (!Number.isFinite(numeric) || (logarithmic && numeric <= 0)) {
            return null;
        }
        return logarithmic ? Math.log(numeric) : numeric;
    }

    function normalizeValue(value, parameter) {
        const logarithmic = Boolean(parameter.log);
        const lower = transformValue(parameter.current_min, logarithmic);
        const upper = transformValue(parameter.current_max, logarithmic);
        const transformed = transformValue(value, logarithmic);
        if (lower === null || upper === null || transformed === null) {
            return null;
        }
        const width = upper - lower;
        return Math.abs(width) <= 1e-8
            ? 0.5
            : (transformed - lower) / width;
    }

    function numericDomain(parameter, selection) {
        const minimum = normalizeValue(selection.min, parameter);
        const maximum = normalizeValue(selection.max, parameter);
        if (minimum === null || maximum === null) {
            return null;
        }
        return [
            Math.min(-SLIDER_EXTENSION, minimum - SLIDER_EXTENSION),
            Math.max(1 + SLIDER_EXTENSION, maximum + SLIDER_EXTENSION),
        ];
    }

    function equalValues(left, right) {
        return left === right || JSON.stringify(left) === JSON.stringify(right);
    }

    function markerPosition(parameter, selection, value) {
        if (parameter.parameter_type === "integer" || parameter.parameter_type === "float") {
            const normalized = normalizeValue(value, parameter);
            const domain = numericDomain(parameter, selection);
            if (normalized === null || domain === null || domain[1] === domain[0]) {
                return null;
            }
            return 100 * (normalized - domain[0]) / (domain[1] - domain[0]);
        }

        const choices = Array.isArray(parameter.choices) ? parameter.choices : [];
        const choiceIndex = choices.findIndex(function (choice) {
            return equalValues(choice, value);
        });
        if (choiceIndex < 0) {
            return null;
        }
        return choices.length === 1 ? 50 : 100 * choiceIndex / (choices.length - 1);
    }

    function formatValue(value) {
        if (typeof value === "number" && Number.isFinite(value)) {
            return Number(value.toPrecision(6)).toString();
        }
        return String(value);
    }

    function markerComponent(trial, value, position) {
        const rank = trial.rank === undefined || trial.rank === null ? "?" : trial.rank;
        const trialNumber = trial.trial_number === undefined || trial.trial_number === null
            ? "?"
            : trial.trial_number;
        return {
            namespace: "dash_html_components",
            type: "Span",
            props: {
                children: null,
                className: "planner-top-trial-marker",
                style: {left: `${position.toFixed(6)}%`},
                title: `Rank ${rank} · Trial ${trialNumber} · ${formatValue(value)}`,
                "data-rank": String(rank === "?" ? "" : rank),
            },
        };
    }

    function markersForParameter(parameter, selection, trials) {
        const selected = selection[parameter.name];
        if (!selected) {
            return [];
        }
        const markers = [];
        trials.forEach(function (trial) {
            const configuration = trial && trial.configuration;
            if (!configuration || configuration[parameter.name] === undefined ||
                    configuration[parameter.name] === null) {
                return;
            }
            const value = configuration[parameter.name];
            const position = markerPosition(parameter, selected, value);
            if (position === null || !Number.isFinite(position) ||
                    position < -1e-9 || position > 100 + 1e-9) {
                return;
            }
            markers.push(markerComponent(trial, value, position));
        });
        return markers;
    }

    const dashClientside = window.dash_clientside || {};
    window.dash_clientside = Object.assign(dashClientside, {
        hpoPlanning: Object.assign({}, dashClientside.hpoPlanning, {
            updateTopTrialMarkers: function (
                clicks,
                showTopN,
                selection,
                activeValues,
                buttonIds,
                layerIds,
                metadata,
                coverageData
            ) {
                if (!Array.isArray(metadata) || !metadata.length || !selection ||
                        !coverageData || !Array.isArray(coverageData.trials) ||
                        !Array.isArray(buttonIds) || !buttonIds.length) {
                    return [
                        window.dash_clientside.no_update,
                        window.dash_clientside.no_update,
                        window.dash_clientside.no_update,
                    ];
                }

                const validIndices = new Set(metadata.map(function (parameter) {
                    return Number(parameter.index);
                }));
                const visible = new Set();
                buttonIds.forEach(function (buttonId, index) {
                    const parameterIndex = Number(buttonId.index);
                    if (activeValues && activeValues[index] && validIndices.has(parameterIndex)) {
                        visible.add(parameterIndex);
                    }
                });

                const trigger = triggeredId();
                if (trigger && typeof trigger === "object" &&
                        trigger.type === "hpo-planner-top-trials") {
                    const parameterIndex = Number(trigger.index);
                    const buttonIndex = buttonIds.findIndex(function (buttonId) {
                        return Number(buttonId.index) === parameterIndex;
                    });
                    if (!clicks || buttonIndex < 0 || !clicks[buttonIndex]) {
                        return [
                            window.dash_clientside.no_update,
                            window.dash_clientside.no_update,
                            window.dash_clientside.no_update,
                        ];
                    }
                    if (visible.has(parameterIndex)) {
                        visible.delete(parameterIndex);
                    } else if (validIndices.has(parameterIndex)) {
                        visible.add(parameterIndex);
                    }
                }

                const active = buttonIds.map(function (buttonId) {
                    return visible.has(Number(buttonId.index));
                });
                const parameterLookup = new Map(metadata.map(function (parameter) {
                    return [Number(parameter.index), parameter];
                }));
                const trials = coverageData.trials.slice(0, topTrialCount(showTopN));
                const layers = (layerIds || []).map(function (layerId) {
                    const index = Number(layerId.index);
                    const parameter = parameterLookup.get(index);
                    if (!parameter || !visible.has(index)) {
                        return [];
                    }
                    return markersForParameter(parameter, selection, trials);
                });
                const titles = buttonIds.map(function (buttonId, index) {
                    const parameter = parameterLookup.get(Number(buttonId.index));
                    const action = active[index] ? "Hide" : "Show";
                    return `${action} top-trial positions for ${parameter ? parameter.name : "parameter"}`;
                });
                return [active, titles, layers];
            },
        }),
    });
})();
