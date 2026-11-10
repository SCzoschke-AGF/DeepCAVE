/* Trial data is downloaded once; table/plot controls never upload it back to Dash. */
(function () {
    "use strict";
    const client = window.dash_clientside = window.dash_clientside || {};
    function finite(value) {
        return value !== null && value !== undefined && typeof value !== "boolean" &&
            Number.isFinite(Number(value));
    }
    function text(value) {
        return value === null ? "None" : value === true ? "True" : value === false ? "False" : String(value);
    }
    client.candidateExplorer = {
        objectives: function (data, current) {
            const objectives = (data || {}).objectives || {};
            const options = Object.keys(objectives).map(key => ({label: objectives[key].label, value: key}));
            return [options, current == null ? Object.keys(objectives) : current.filter(key => key in objectives)];
        },
        table: function (data, metrics, presentation) {
            data = data || {};
            const source = data.rows || [];
            const parameters = Array.from(new Set(source.flatMap(row => Object.keys(row.parameters)))).sort();
            const numeric = new Set(parameters.filter(name => source.every(row =>
                !(name in row.parameters) || typeof row.parameters[name] === "number")));
            const columns = [{name: "Study", id: "study_name", type: "text"},
                {name: "Trial", id: "trial_number", type: "numeric"},
                {name: "Duration (s)", id: "duration", type: "numeric", format: presentation.numberFormat}];
            (metrics || []).forEach(function (key) {
                if ((data.objectives || {})[key]) columns.push({name: data.objectives[key].label,
                    id: key, type: "numeric", format: presentation.numberFormat});
            });
            parameters.forEach(name => columns.push({name: name, id: "param:" + name,
                type: numeric.has(name) ? "numeric" : "text"}));
            const rows = source.filter(row => row.state === "COMPLETE").map(function (row) {
                const flat = {};
                ["id", "study_name", "trial_number", "trial_id", "state", "duration"].forEach(key => flat[key] = row[key]);
                Object.assign(flat, row.metrics);
                Object.keys(row.parameters).forEach(name => flat["param:" + name] =
                    numeric.has(name) ? row.parameters[name] : text(row.parameters[name]));
                return flat;
            });
            const tooltips = rows.map(row => Object.fromEntries(columns.map(column =>
                [column.id, {value: column.id in row ? text(row[column.id]) : "", type: "text"}])));
            return [{rows: rows, tooltips: tooltips}, columns];
        },
        visibleIds: function (ids, previous) {
            const next = ids == null ? null : ids.slice().sort();
            return JSON.stringify(next) === JSON.stringify(previous) ? client.no_update : next;
        },
        scatter: function (visibleIds, data, state, pareto, elites, fraction, presentation) {
            data = data || {};
            state = state || {};
            const keymap = data.keymap || {};
            const token = state.value || "objective:0";
            const objective = token in keymap ? keymap[token] : Object.values(keymap)[0];
            const ids = visibleIds == null ? null : new Set(visibleIds);
            const rows = (data.rows || []).filter(row => row.state === "COMPLETE" &&
                (!ids || ids.has(row.id)) && finite(row.duration) && row.duration >= 0 &&
                finite(row.metrics[objective]));
            const direction = (objective || "").startsWith("user_attr:") ? state.direction : null;
            const directions = new Set(rows.map(row => direction || row.metric_info[objective].optimize));
            const maximize = directions.has("upper");
            const x = row => row.duration;
            const y = row => row.metrics[objective];
            const indices = rows.map((row, i) => i);
            fraction = Number(fraction || 0.05);
            const elite = new Set(elites && directions.size === 1 ? indices.slice().sort((a, b) =>
                (maximize ? y(rows[b]) - y(rows[a]) : y(rows[a]) - y(rows[b])) || a - b
            ).slice(0, Math.max(1, Math.ceil(rows.length * fraction))) : []);
            const layout = JSON.parse(JSON.stringify(presentation.scatter.layout));
            delete layout.annotations;
            layout.uirevision = JSON.stringify([objective || null, Boolean(elites), elites ? fraction : null]);
            layout.yaxis.title.text = rows.length ? rows[0].metric_info[objective].label : objective || "Objective";
            const traces = [];
            function note(message, height) {
                layout.annotations = layout.annotations || [];
                layout.annotations.push({text: message, x: 0.5, y: height, xref: "paper", yref: "paper", showarrow: false, font: {size: 14}});
            }
            if (rows.length) {
                traces.push({type: "scattergl", x: rows.map(x), y: rows.map(y),
                    customdata: rows.map((row, i) => [row.id, row.study_name, row.trial_number, row.color, elite.has(i)]),
                    mode: "markers", name: "Trials", hoverinfo: "none",
                    marker: {color: rows.map(() => "#737373"), size: rows.map(() => 6), opacity: rows.map(() => 1)}});
                if (elites && directions.size !== 1) note("Elites unavailable: studies use different directions.", 1.14);
                if (pareto && directions.size === 1) {
                    const ordered = indices.slice().sort((a, b) => x(rows[a]) - x(rows[b]) ||
                        (maximize ? y(rows[b]) - y(rows[a]) : y(rows[a]) - y(rows[b])));
                    let best = Infinity;
                    const frontier = [];
                    for (let start = 0; start < ordered.length;) {
                        let end = start + 1;
                        while (end < ordered.length && x(rows[ordered[end]]) === x(rows[ordered[start]])) end++;
                        const value = (maximize ? -1 : 1) * y(rows[ordered[start]]);
                        if (value < best) {
                            for (let i = start; i < end; i++) {
                                if ((maximize ? -1 : 1) * y(rows[ordered[i]]) === value) frontier.push(rows[ordered[i]]);
                            }
                            best = value;
                        }
                        start = end;
                    }
                    traces.push({type: "scatter", x: frontier.map(x), y: frontier.map(y), mode: "lines",
                        line: {color: "#64748b", width: 1, dash: "dot"}, hoverinfo: "skip", name: "Pareto frontier"});
                } else if (pareto) note("Pareto frontier unavailable: studies use different directions.", 1.06);
            } else note("No visible trials with a valid objective and duration.", 0.5);
            if (elite.size) {
                [["xaxis", x], ["yaxis", y]].forEach(function (axis) {
                    const values = Array.from(elite, i => axis[1](rows[i]));
                    const low = Math.min(...values), high = Math.max(...values);
                    const padding = (high - low) * 0.05 || Math.max(Math.abs(low) * 0.05, 0.05);
                    layout[axis[0]].range = [axis[0] === "xaxis" ? Math.max(0, low - padding) : low - padding, high + padding];
                    layout[axis[0]].autorange = false;
                });
            }
            return {data: traces, layout: layout};
        }
    };
}());
