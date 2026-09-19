(function () {
    "use strict";

    const MASK_COLOR = "rgba(100, 116, 139, 0.42)";

    function sameValue(left, right) {
        if (left === right) {
            return true;
        }
        return typeof left === typeof right && left !== null && right !== null &&
            typeof left === "object" && JSON.stringify(left) === JSON.stringify(right);
    }

    function transform(value, logarithmic) {
        const numeric = Number(value);
        if (!Number.isFinite(numeric) || (logarithmic && numeric <= 0)) {
            return null;
        }
        return logarithmic ? Math.log(numeric) : numeric;
    }

    function coerceNumeric(candidate, parameter) {
        if (!candidate || typeof candidate !== "object") {
            return null;
        }
        let minimum = Number(candidate.min);
        let maximum = Number(candidate.max);
        if (!Number.isFinite(minimum) || !Number.isFinite(maximum) ||
                transform(minimum, Boolean(parameter.log)) === null ||
                transform(maximum, Boolean(parameter.log)) === null) {
            return null;
        }
        if (minimum > maximum) {
            const temporary = minimum;
            minimum = maximum;
            maximum = temporary;
        }
        return {min: minimum, max: maximum};
    }

    function selectionFor(metadata, stored) {
        const selection = {};
        (metadata || []).forEach(function (parameter) {
            const candidate = stored && stored[parameter.name];
            if (parameter.parameter_type === "integer" || parameter.parameter_type === "float") {
                selection[parameter.name] = coerceNumeric(candidate, parameter) || {
                    min: parameter.current_min,
                    max: parameter.current_max,
                };
            } else {
                const selected = candidate && Array.isArray(candidate.choices)
                    ? candidate.choices
                    : parameter.choices || [];
                selection[parameter.name] = {
                    choices: (parameter.choices || []).filter(function (choice) {
                        return selected.some(function (value) {
                            return sameValue(value, choice);
                        });
                    }),
                };
            }
        });
        return selection;
    }

    function mergeSegments(segments) {
        const merged = [];
        segments.slice().sort(function (left, right) {
            return left[0] - right[0];
        }).forEach(function (segment) {
            const start = segment[0];
            const end = segment[1];
            if (!(end > start)) {
                return;
            }
            const previous = merged[merged.length - 1];
            if (previous && start <= previous[1] + 1e-9) {
                previous[1] = Math.max(previous[1], end);
            } else {
                merged.push([start, end]);
            }
        });
        return merged;
    }

    function complement(included, domain) {
        const excluded = [];
        let cursor = domain[0];
        mergeSegments(included).forEach(function (segment) {
            const start = Math.max(domain[0], segment[0]);
            const end = Math.min(domain[1], segment[1]);
            if (start > cursor) {
                excluded.push([cursor, start]);
            }
            cursor = Math.max(cursor, end);
        });
        if (cursor < domain[1]) {
            excluded.push([cursor, domain[1]]);
        }
        return excluded;
    }

    function continuousSegments(parameter, selection) {
        const domain = [Number(parameter.current_min), Number(parameter.current_max)];
        const start = Math.max(domain[0], Number(selection.min));
        const end = Math.min(domain[1], Number(selection.max));
        return [end > start ? [[start, end]] : [], domain];
    }

    function binnedSegments(axes, parameter, selection) {
        const included = [];
        (axes || []).forEach(function (axis, index) {
            if (parameter.parameter_type === "integer" || parameter.parameter_type === "float") {
                if ((axis.kind || "discrete") === "interval") {
                    const lower = Number(axis.lower);
                    const upper = Number(axis.upper);
                    const selectedLower = Math.max(lower, Number(selection.min));
                    const selectedUpper = Math.min(upper, Number(selection.max));
                    if (!(selectedUpper > selectedLower)) {
                        return;
                    }
                    const transformedLower = transform(lower, Boolean(parameter.log));
                    const transformedUpper = transform(upper, Boolean(parameter.log));
                    const transformedSelectedLower = transform(
                        selectedLower, Boolean(parameter.log)
                    );
                    const transformedSelectedUpper = transform(
                        selectedUpper, Boolean(parameter.log)
                    );
                    const width = transformedUpper - transformedLower;
                    if (transformedLower === null || transformedUpper === null ||
                            transformedSelectedLower === null ||
                            transformedSelectedUpper === null || Math.abs(width) <= 1e-12) {
                        return;
                    }
                    included.push([
                        index - 0.5 + (transformedSelectedLower - transformedLower) / width,
                        index - 0.5 + (transformedSelectedUpper - transformedLower) / width,
                    ]);
                } else {
                    const value = Number(
                        axis.value === undefined ? axis.coordinate : axis.value
                    );
                    if (Number(selection.min) <= value && value <= Number(selection.max)) {
                        included.push([index - 0.5, index + 0.5]);
                    }
                }
            } else {
                const value = axis.value === undefined ? axis.coordinate : axis.value;
                if ((selection.choices || []).some(function (choice) {
                    return sameValue(choice, value);
                })) {
                    included.push([index - 0.5, index + 0.5]);
                }
            }
        });
        return [mergeSegments(included), [-0.5, (axes || []).length - 0.5]];
    }

    function maskShapes(xGeometry, yGeometry) {
        const shapes = [];
        const base = {
            type: "rect",
            xref: "x",
            yref: "y",
            fillcolor: MASK_COLOR,
            line: {width: 0},
            layer: "above",
        };
        complement(xGeometry[0], xGeometry[1]).forEach(function (segment) {
            shapes.push(Object.assign({}, base, {
                x0: segment[0], x1: segment[1],
                y0: yGeometry[1][0], y1: yGeometry[1][1],
            }));
        });
        complement(yGeometry[0], yGeometry[1]).forEach(function (ySegment) {
            xGeometry[0].forEach(function (xSegment) {
                shapes.push(Object.assign({}, base, {
                    x0: xSegment[0], x1: xSegment[1],
                    y0: ySegment[0], y1: ySegment[1],
                }));
            });
        });
        return shapes;
    }

    function shapesForFigure(figure, planningState) {
        const mask = figure && figure.layout && figure.layout.meta &&
            figure.layout.meta.deepcave_planning_mask;
        if (!mask || !mask.run_id || !mask.objective_key || !Array.isArray(mask.metadata) ||
                mask.metadata.length !== 2) {
            return null;
        }
        const runs = planningState && planningState.runs;
        const scopes = runs && runs[mask.run_id];
        const selection = selectionFor(
            mask.metadata,
            scopes && scopes[mask.objective_key]
        );
        const geometries = mask.metadata.map(function (parameter, index) {
            if (mask.kind === "continuous" ||
                    (mask.kind === "scatter" &&
                        (parameter.parameter_type === "integer" ||
                         parameter.parameter_type === "float"))) {
                return continuousSegments(parameter, selection[parameter.name]);
            }
            const axes = index === 0 ? mask.axis_1 : mask.axis_2;
            return binnedSegments(axes, parameter, selection[parameter.name]);
        });
        return maskShapes(geometries[0], geometries[1]);
    }

    function updateFigure(figure, planningState) {
        const shapes = shapesForFigure(figure, planningState);
        if (shapes === null) {
            return window.dash_clientside.no_update;
        }
        if (JSON.stringify((figure.layout && figure.layout.shapes) || []) ===
                JSON.stringify(shapes)) {
            return window.dash_clientside.no_update;
        }
        return Object.assign({}, figure, {
            layout: Object.assign({}, figure.layout, {shapes: shapes}),
        });
    }

    const dashClientside = window.dash_clientside || {};
    window.dash_clientside = Object.assign(dashClientside, {
        twoDimensionalPlanning: Object.assign({}, dashClientside.twoDimensionalPlanning, {
            applyMasks: function (planningState, trials, enrichment, objective) {
                return [
                    updateFigure(trials, planningState),
                    updateFigure(enrichment, planningState),
                    updateFigure(objective, planningState),
                ];
            },
            applyMountedMasks: function (planningState, revision) {
                let updated = false;
                [
                    "two_dimensional_analysis-trials_graph-output",
                    "two_dimensional_analysis-elite_rate_graph-output",
                    "two_dimensional_analysis-objective_graph-output",
                ].forEach(function (id) {
                    const container = document.getElementById(id);
                    const graph = container && container.classList.contains("js-plotly-plot")
                        ? container
                        : container && container.querySelector(".js-plotly-plot");
                    if (!graph || !graph.layout || !window.Plotly) {
                        return;
                    }
                    const shapes = shapesForFigure({layout: graph.layout}, planningState);
                    if (shapes === null ||
                            JSON.stringify(graph.layout.shapes || []) === JSON.stringify(shapes)) {
                        return;
                    }
                    window.Plotly.relayout(graph, {shapes: shapes});
                    updated = true;
                });
                return updated ? (Number(revision) || 0) + 1 : window.dash_clientside.no_update;
            },
        }),
    });
})();
