(function () {
    "use strict";

    const NO_UPDATE = window.dash_clientside.no_update;

    function component(type, props) {
        return {namespace: "dash_html_components", type: type, props: props};
    }

    function percentage(value, domain) {
        return 100 * (Number(value) - domain[0]) / (domain[1] - domain[0]);
    }

    function percentStyle(value) {
        return `${value.toFixed(6)}%`;
    }

    function formatValue(value) {
        if (typeof value === "number" && Number.isFinite(value)) {
            return Number(value.toPrecision(6)).toString();
        }
        return value === null || value === undefined ? "None" : String(value);
    }

    function referenceComponent(domain) {
        const left = percentage(0, domain);
        const right = percentage(1, domain);
        return component("Span", {
            children: null,
            className: "planner-current-reference-segment",
            style: {left: percentStyle(left), width: percentStyle(right - left)},
        });
    }

    function extensionComponents(value, domain) {
        const components = [];
        [
            [Number(value[0]), Math.min(Number(value[1]), 0)],
            [Math.max(Number(value[0]), 1), Number(value[1])],
        ].forEach(function (segment) {
            if (!(segment[1] > segment[0])) {
                return;
            }
            const left = percentage(segment[0], domain);
            const right = percentage(segment[1], domain);
            components.push(component("Span", {
                children: null,
                className: "planner-extension-segment",
                style: {left: percentStyle(left), width: percentStyle(right - left)},
            }));
        });
        return components;
    }

    function q90UnselectedComponents(value) {
        const minimum = Math.min(100, Math.max(0, 100 * Number(value[0])));
        const maximum = Math.min(100, Math.max(0, 100 * Number(value[1])));
        return [
            component("Span", {
                children: null,
                className: "planner-unselected-segment planner-q90-unselected-segment",
                style: {left: "0", width: percentStyle(minimum)},
            }),
            component("Span", {
                children: null,
                className: "planner-unselected-segment planner-q90-unselected-segment",
                style: {left: percentStyle(maximum), width: percentStyle(100 - maximum)},
            }),
        ];
    }

    function eliteTrack(elite, domain) {
        const minimum = elite.minimum_position;
        const q10 = elite.q10_position;
        const median = elite.median_position;
        const q90 = elite.q90_position;
        const maximum = elite.maximum_position;
        const children = [];
        let title = `${elite.group}: no active trials`;

        if (q10 !== null && q10 !== undefined && median !== null && median !== undefined &&
                q90 !== null && q90 !== undefined) {
            const q10Percent = percentage(q10, domain);
            const medianPercent = percentage(median, domain);
            const q90Percent = percentage(q90, domain);
            if (minimum !== null && minimum !== undefined &&
                    maximum !== null && maximum !== undefined) {
                const minimumPercent = percentage(minimum, domain);
                const maximumPercent = percentage(maximum, domain);
                children.push(component("Span", {
                    children: null,
                    className: "planner-elite-whisker",
                    style: {
                        left: percentStyle(minimumPercent),
                        width: percentStyle(Math.max(0, maximumPercent - minimumPercent)),
                        backgroundColor: elite.color,
                    },
                }));
            }
            children.push(component("Span", {
                children: null,
                className: "planner-elite-band",
                style: {
                    left: percentStyle(q10Percent),
                    width: percentStyle(Math.max(0, q90Percent - q10Percent)),
                    backgroundColor: elite.color,
                },
            }));
            children.push(component("Span", {
                children: null,
                className: "planner-elite-marker",
                style: {left: percentStyle(medianPercent), backgroundColor: elite.color},
            }));
            title = `${elite.group} min–max: ${formatValue(elite.minimum)}–` +
                `${formatValue(elite.maximum)}; Q10–Q90: ${formatValue(elite.q10)}–` +
                `${formatValue(elite.q90)}; median: ${formatValue(elite.median)} (n=${elite.n})`;
        }
        return component("Div", {
            children: children,
            title: title,
            className: "planner-elite-track",
        });
    }

    function parameterForRange(metadata, rangeId) {
        if (!Array.isArray(metadata) || !rangeId) {
            return null;
        }
        return metadata.find(function (parameter) {
            return Number(parameter.index) === Number(rangeId.index) &&
                parameter.planning_context === rangeId.context;
        }) || null;
    }

    const dashClientside = window.dash_clientside || {};
    window.dash_clientside = Object.assign(dashClientside, {
        hpoPlanning: Object.assign({}, dashClientside.hpoPlanning, {
            updateRangeOverlays: function (minimum, maximum, value, selection, rangeId, metadata) {
                const domain = [Number(minimum), Number(maximum)];
                const parameter = parameterForRange(metadata, rangeId);
                if (!parameter || !Array.isArray(parameter.elite_positions) ||
                        !Array.isArray(value) || value.length !== 2 ||
                        !domain.every(Number.isFinite) || !(domain[1] > domain[0]) ||
                        !value.map(Number).every(Number.isFinite)) {
                    return [NO_UPDATE, NO_UPDATE, NO_UPDATE, NO_UPDATE, NO_UPDATE];
                }
                const selected = selection && selection[parameter.name];
                const rangeSummary = selected && selected.min !== undefined &&
                    selected.max !== undefined
                    ? `original: ${formatValue(parameter.current_min)}–` +
                        `${formatValue(parameter.current_max)} | current: ` +
                        `${formatValue(selected.min)}–${formatValue(selected.max)}`
                    : NO_UPDATE;
                return [
                    referenceComponent(domain),
                    extensionComponents(value, domain),
                    parameter.elite_positions.map(function (elite) {
                        return eliteTrack(elite, domain);
                    }),
                    rangeSummary,
                    q90UnselectedComponents(value),
                ];
            },
        }),
    });
})();
