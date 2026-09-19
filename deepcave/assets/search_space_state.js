(function () {
    "use strict";

    const VERSION = 2;
    const EXTENSION = 0.25;
    const NO_UPDATE = window.dash_clientside.no_update;

    function deepEqual(left, right) {
        if (left === right) {
            return true;
        }
        if (typeof left !== typeof right || left === null || right === null) {
            return false;
        }
        if (Array.isArray(left) || Array.isArray(right)) {
            return Array.isArray(left) && Array.isArray(right) &&
                left.length === right.length && left.every(function (value, index) {
                    return deepEqual(value, right[index]);
                });
        }
        if (typeof left === "object") {
            const leftKeys = Object.keys(left);
            const rightKeys = Object.keys(right);
            return leftKeys.length === rightKeys.length && leftKeys.every(function (key) {
                return Object.prototype.hasOwnProperty.call(right, key) &&
                    deepEqual(left[key], right[key]);
            });
        }
        return false;
    }

    function roundHalfEven(value) {
        const floor = Math.floor(value);
        const fraction = value - floor;
        if (Math.abs(fraction - 0.5) <= Number.EPSILON * Math.max(1, Math.abs(value))) {
            return floor % 2 === 0 ? floor : floor + 1;
        }
        return Math.round(value);
    }

    function transform(value, logarithmic) {
        const numeric = Number(value);
        if (!Number.isFinite(numeric) || (logarithmic && numeric <= 0)) {
            return null;
        }
        return logarithmic ? Math.log(numeric) : numeric;
    }

    function normalize(value, parameter) {
        const lower = transform(parameter.current_min, Boolean(parameter.log));
        const upper = transform(parameter.current_max, Boolean(parameter.log));
        const transformed = transform(value, Boolean(parameter.log));
        if (lower === null || upper === null || transformed === null) {
            return null;
        }
        return Math.abs(upper - lower) <= 1e-12
            ? 0.5
            : (transformed - lower) / (upper - lower);
    }

    function denormalize(position, parameter) {
        const lower = transform(parameter.current_min, Boolean(parameter.log));
        const upper = transform(parameter.current_max, Boolean(parameter.log));
        const numericPosition = Number(position);
        if (lower === null || upper === null || !Number.isFinite(numericPosition)) {
            return null;
        }
        const value = lower + numericPosition * (upper - lower);
        return parameter.log ? Math.exp(value) : value;
    }

    function snap(value, parameter) {
        let numeric = Number(value);
        if (!Number.isFinite(numeric)) {
            return null;
        }
        const step = Number(parameter.step);
        if (parameter.step !== null && parameter.step !== undefined &&
                Number.isFinite(step) && step > 0) {
            const lower = Number(parameter.current_min);
            numeric = lower + roundHalfEven((numeric - lower) / step) * step;
        }
        if (parameter.parameter_type === "integer") {
            return roundHalfEven(numeric);
        }
        return Number(numeric.toPrecision(12));
    }

    function coerce(minimum, maximum, parameter) {
        let lower = snap(minimum, parameter);
        let upper = snap(maximum, parameter);
        if (lower === null || upper === null ||
                transform(lower, Boolean(parameter.log)) === null ||
                transform(upper, Boolean(parameter.log)) === null) {
            return null;
        }
        if (lower > upper) {
            const temporary = lower;
            lower = upper;
            upper = temporary;
        }
        return {min: lower, max: upper};
    }

    function typedIncludes(values, candidate) {
        return values.some(function (value) {
            return deepEqual(value, candidate);
        });
    }

    function initialSelection(metadata) {
        const selection = {};
        (metadata || []).forEach(function (parameter) {
            if (parameter.parameter_type === "integer" || parameter.parameter_type === "float") {
                selection[parameter.name] = {
                    min: parameter.current_min,
                    max: parameter.current_max,
                };
            } else {
                selection[parameter.name] = {choices: (parameter.choices || []).slice()};
            }
        });
        return selection;
    }

    function reconcile(metadata, stored) {
        const selection = initialSelection(metadata);
        if (!stored || typeof stored !== "object") {
            return selection;
        }
        (metadata || []).forEach(function (parameter) {
            let candidate = stored[parameter.name];
            if (Array.isArray(candidate)) {
                candidate = (parameter.parameter_type === "integer" ||
                    parameter.parameter_type === "float") && candidate.length === 2
                    ? {min: candidate[0], max: candidate[1]}
                    : {choices: candidate};
            }
            if (!candidate || typeof candidate !== "object") {
                return;
            }
            if (parameter.parameter_type === "integer" || parameter.parameter_type === "float") {
                const coerced = coerce(candidate.min, candidate.max, parameter);
                if (coerced !== null) {
                    selection[parameter.name] = coerced;
                }
            } else if (Array.isArray(candidate.choices)) {
                selection[parameter.name] = {
                    choices: (parameter.choices || []).filter(function (choice) {
                        return typedIncludes(candidate.choices, choice);
                    }),
                };
            }
        });
        return selection;
    }

    function sourceForType(type) {
        if (typeof type !== "string") {
            return null;
        }
        if (type.indexOf("hpo-planner-") === 0 || type === "hpo-planning-reset") {
            return "hpo-planning";
        }
        if (type.indexOf("two-dimensional-planning-") === 0) {
            return "two-dimensional-analysis";
        }
        return null;
    }

    function parseTriggeredId(item) {
        if (!item || !item.prop_id) {
            return null;
        }
        const raw = item.prop_id.slice(0, item.prop_id.lastIndexOf("."));
        try {
            return JSON.parse(raw);
        } catch (_error) {
            return null;
        }
    }

    function contextForId(contexts, id) {
        const source = sourceForType(id && id.type);
        return (contexts || []).find(function (context) {
            return context && context.source === source && context.run_id &&
                context.objective_key && context.planning_context === id.context &&
                Array.isArray(context.metadata);
        });
    }

    function storedSelection(state, context) {
        const runs = state && state.runs && typeof state.runs === "object" ? state.runs : {};
        const scopes = runs[context.run_id];
        return scopes && typeof scopes === "object" ? scopes[context.objective_key] : null;
    }

    function setStoredSelection(state, context, selection) {
        const scopes = Object.assign({}, state.runs[context.run_id] || {});
        scopes[context.objective_key] = selection;
        state.runs[context.run_id] = scopes;
        state.last_change = {
            run_id: context.run_id,
            objective_key: context.objective_key,
        };
    }

    function mergePersistedState(state, persisted) {
        if (!persisted || persisted.version !== VERSION ||
                !persisted.runs || typeof persisted.runs !== "object") {
            return false;
        }
        let changed = false;
        Object.keys(persisted.runs).forEach(function (runId) {
            const persistedScopes = persisted.runs[runId];
            if (!persistedScopes || typeof persistedScopes !== "object") {
                return;
            }
            const currentScopes = Object.assign({}, state.runs[runId] || {});
            let runChanged = false;
            Object.keys(persistedScopes).forEach(function (objectiveKey) {
                if (!Object.prototype.hasOwnProperty.call(currentScopes, objectiveKey) &&
                        persistedScopes[objectiveKey] &&
                        typeof persistedScopes[objectiveKey] === "object") {
                    currentScopes[objectiveKey] = persistedScopes[objectiveKey];
                    changed = true;
                    runChanged = true;
                }
            });
            if (runChanged) {
                state.runs[runId] = currentScopes;
            }
        });
        return changed;
    }

    function parameterForId(context, id) {
        return (context.metadata || []).find(function (parameter) {
            return Number(parameter.index) === Number(id.index);
        });
    }

    function categorySelection(value, parameter) {
        if (!Array.isArray(value)) {
            return null;
        }
        return {
            choices: (parameter.choices || []).filter(function (choice) {
                return typedIncludes(value, choice);
            }),
        };
    }

    function changedSelection(id, value, parameter, current) {
        if (id.type === "hpo-planner-range" ||
                id.type === "two-dimensional-planning-range") {
            if (!Array.isArray(value) || value.length !== 2) {
                return null;
            }
            const minimum = denormalize(value[0], parameter);
            const maximum = denormalize(value[1], parameter);
            return coerce(minimum, maximum, parameter);
        }
        if (id.type === "hpo-planner-min") {
            return coerce(value, current.max, parameter);
        }
        if (id.type === "hpo-planner-max") {
            return coerce(current.min, value, parameter);
        }
        if (id.type === "hpo-planner-categories" ||
                id.type === "two-dimensional-planning-categories") {
            return categorySelection(value, parameter);
        }
        return null;
    }

    function numericView(parameter, selection) {
        const minimum = normalize(selection.min, parameter);
        const maximum = normalize(selection.max, parameter);
        if (minimum === null || maximum === null) {
            return null;
        }
        return {
            value: [minimum, maximum],
            min: Math.min(-EXTENSION, minimum - EXTENSION),
            max: Math.max(1 + EXTENSION, maximum + EXTENSION),
        };
    }

    function controlValue(ids, contexts, state, property) {
        return (ids || []).map(function (id) {
            const context = contextForId(contexts, id);
            if (!context) {
                return NO_UPDATE;
            }
            const parameter = parameterForId(context, id);
            if (!parameter) {
                return NO_UPDATE;
            }
            const selection = reconcile(context.metadata, storedSelection(state, context));
            const selected = selection[parameter.name];
            if (property === "categories") {
                return selected && Array.isArray(selected.choices)
                    ? selected.choices.slice()
                    : NO_UPDATE;
            }
            if (property === "minimum") {
                return selected ? selected.min : NO_UPDATE;
            }
            if (property === "maximum") {
                return selected ? selected.max : NO_UPDATE;
            }
            const view = selected ? numericView(parameter, selected) : null;
            return view ? view[property] : NO_UPDATE;
        });
    }

    function reducePlanningEvents(
        _hpoRanges,
        _hpoMinimums,
        _hpoMaximums,
        _hpoCategories,
        _hpoResetClicks,
        _twoRanges,
        _twoCategories,
        _twoResetClicks,
        contexts,
        persistedState,
        hpoRangeIds,
        hpoMinimumIds,
        hpoMaximumIds,
        hpoCategoryIds,
        twoRangeIds,
        twoCategoryIds,
        previousState
    ) {
        const valid = previousState && previousState.version === VERSION &&
            previousState.runs && typeof previousState.runs === "object";
        const state = valid ? {
            version: VERSION,
            revision: Number(previousState.revision) || 0,
            runs: Object.assign({}, previousState.runs),
            event_sequences: Object.assign({}, previousState.event_sequences || {}),
            last_change: previousState.last_change || null,
        } : {
            version: VERSION,
            revision: 0,
            runs: {},
            event_sequences: {},
            last_change: null,
        };
        let changed = false;
        const callbackContext = window.dash_clientside.callback_context;
        const triggered = callbackContext && Array.isArray(callbackContext.triggered)
            ? callbackContext.triggered
            : [];
        const mountedContexts = new Set(triggered.reduce(function (values, event) {
            const id = parseTriggeredId(event);
            const context = event && event.value;
            if (id && id.type === "planned-search-space-context" && context &&
                    context.planning_context) {
                values.push(context.planning_context);
            }
            return values;
        }, []));

        triggered.forEach(function (event) {
            const id = parseTriggeredId(event);
            if (event && event.prop_id === "planned-search-space-hydration.data") {
                if (mergePersistedState(state, persistedState)) {
                    changed = true;
                }
                return;
            }
            if (id && id.type === "planned-search-space-context") {
                const mountedContext = event.value;
                if (mountedContext && mountedContext.run_id &&
                        mountedContext.objective_key &&
                        Array.isArray(mountedContext.metadata) &&
                        storedSelection(state, mountedContext)) {
                    const stored = storedSelection(state, mountedContext);
                    const reconciled = reconcile(
                        mountedContext.metadata,
                        stored
                    );
                    const merged = Object.assign(
                        {},
                        stored,
                        reconciled
                    );
                    if (!deepEqual(merged, stored)) {
                        setStoredSelection(state, mountedContext, merged);
                        changed = true;
                    }
                }
                return;
            }
            const context = id ? contextForId(contexts, id) : null;
            if (!context) {
                return;
            }
            // Controls mounted together with a new context report their default values as inputs.
            // Hydrate them from the canonical scope instead of committing those defaults.
            if (mountedContexts.has(context.planning_context)) {
                return;
            }
            const currentStored = storedSelection(state, context);
            const current = reconcile(context.metadata, currentStored);
            if (id.type === "hpo-planning-reset" ||
                    id.type === "two-dimensional-planning-reset") {
                // A dynamically mounted button reports n_clicks=0. Only a real click may reset
                // the shared search space; otherwise changing one 2-D feature resets the other.
                if (!event.value) {
                    return;
                }
                const reset = initialSelection(context.metadata);
                if (deepEqual(current, reset)) {
                    return;
                }
                setStoredSelection(
                    state,
                    context,
                    id.type === "hpo-planning-reset"
                        ? reset
                        : Object.assign({}, currentStored || {}, reset)
                );
                changed = true;
                return;
            }

            const parameter = parameterForId(context, id);
            if (!parameter || !current[parameter.name]) {
                return;
            }
            const updated = changedSelection(id, event.value, parameter, current[parameter.name]);
            if (updated === null || deepEqual(updated, current[parameter.name])) {
                return;
            }
            const runSelection = Object.assign({}, currentStored || {});
            runSelection[parameter.name] = updated;
            setStoredSelection(state, context, runSelection);
            changed = true;
        });

        if (changed) {
            state.revision += 1;
        }
        const storeOutput = changed || !valid ? state : NO_UPDATE;
        return [
            storeOutput,
            controlValue(hpoRangeIds, contexts, state, "value"),
            controlValue(hpoRangeIds, contexts, state, "min"),
            controlValue(hpoRangeIds, contexts, state, "max"),
            controlValue(hpoMinimumIds, contexts, state, "minimum"),
            controlValue(hpoMaximumIds, contexts, state, "maximum"),
            controlValue(hpoCategoryIds, contexts, state, "categories"),
            controlValue(twoRangeIds, contexts, state, "value"),
            controlValue(twoRangeIds, contexts, state, "min"),
            controlValue(twoRangeIds, contexts, state, "max"),
            controlValue(twoCategoryIds, contexts, state, "categories"),
        ];
    }

    const dashClientside = window.dash_clientside || {};
    window.dash_clientside = Object.assign(dashClientside, {
        searchSpaceState: Object.assign({}, dashClientside.searchSpaceState, {
            reducePlanningEvents: reducePlanningEvents,
        }),
    });
})();
