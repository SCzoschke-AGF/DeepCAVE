/* Run selection uses the existing Dash field; hover only changes Plotly presentation. */
(function () {
    "use strict";
    const panelId = "cost-over-time-comparison";
    const client = window.dash_clientside = window.dash_clientside || {};
    client.runComparison = {
        distributionMode: function (mode, figure) {
            if (!figure || !figure.data || !figure.layout) return client.no_update;
            const probability = mode !== "count";
            const histnorm = probability ? "probability" : "";
            const title = probability ? "Probability" : "Trials";
            const yaxis = figure.layout.yaxis || {};
            // Copy only this chart's presentation. Samples, bins, tooltips and run state stay intact.
            const updatedAxis = Object.assign({}, yaxis, {
                title: Object.assign({}, yaxis.title, {text: title}),
                tickformat: probability ? ".0%" : "",
                autorange: true,
                uirevision: mode,
            });
            delete updatedAxis.range;
            return Object.assign({}, figure, {
                data: figure.data.map(function (trace) {
                    return trace.type === "histogram"
                        ? Object.assign({}, trace, {histnorm: histnorm}) : trace;
                }),
                layout: Object.assign({}, figure.layout, {yaxis: updatedAxis}),
            });
        },
        selectRun: function (_add, picked, _remove, selected, primary, open) {
            const context = client.callback_context;
            const trigger = context.triggered_id;
            const unchanged = client.no_update;
            if (trigger === panelId + "-add" && _add > 0) {
                return [unchanged, unchanged, !open];
            }
            const runs = Array.from(new Set(selected || []));
            if (trigger === panelId + "-picker" && picked) {
                if (picked !== primary && !runs.includes(picked)) runs.push(picked);
                return [runs, null, false];
            }
            if (trigger && trigger.type === "run-comparison-remove" &&
                    context.triggered.some(function (event) { return Boolean(event.value); }) &&
                    trigger.run !== primary && runs.includes(trigger.run)) {
                return [runs.filter(function (id) { return id !== trigger.run; }), null, false];
            }
            return [unchanged, unchanged, unchanged];
        },
    };

    function rowFor(element) {
        return element && element.closest ? element.closest(".run-comparison__row") : null;
    }

    let highlightedRun = null;
    const boundPlots = new WeakSet();

    function applyHighlight(plot) {
        if (!plot.data || !plot.layout) return;
        const opacity = plot.data.map(function (trace) {
            const base = trace.meta && trace.meta.comparison_opacity;
            return (base === undefined ? 1 : base) *
                (highlightedRun && trace.legendgroup !== highlightedRun ? 0.18 : 1);
        });
        if (opacity.some(function (value, index) { return value !== plot.data[index].opacity; })) {
            window.Plotly.restyle(plot, {opacity: opacity});
        }
        const shapes = {};
        (plot.layout.shapes || []).forEach(function (shape, index) {
            const value = highlightedRun && shape.name !== highlightedRun ? 0.18 : 1;
            if (shape.opacity !== value) shapes["shapes[" + index + "].opacity"] = value;
        });
        if (Object.keys(shapes).length) window.Plotly.relayout(plot, shapes);
    }

    function highlight(runId) {
        highlightedRun = runId;
        if (!window.Plotly) return;
        document.querySelectorAll("#cost_over_time-output .js-plotly-plot").forEach(function (plot) {
            if (!boundPlots.has(plot) && plot.on) {
                // A filter or eye toggle may redraw while the pointer remains over a row.
                plot.on("plotly_afterplot", function () { applyHighlight(plot); });
                boundPlots.add(plot);
            }
            applyHighlight(plot);
        });
    }

    ["mouseover", "focusin"].forEach(function (eventName) {
        document.addEventListener(eventName, function (event) {
            const row = rowFor(event.target);
            if (row && row !== rowFor(event.relatedTarget)) highlight(row.dataset.runId);
        });
    });
    ["mouseout", "focusout"].forEach(function (eventName) {
        document.addEventListener(eventName, function (event) {
            if (rowFor(event.target) && !rowFor(event.relatedTarget)) highlight(null);
        });
    });
})();
