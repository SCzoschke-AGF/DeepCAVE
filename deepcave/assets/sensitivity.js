/* Linked exploration uses separate active configuration, comparison and shared trial states. */
(function () {
    "use strict";
    const client = window.dash_clientside = window.dash_clientside || {};
    const prefix = "sensitivity-";
    const bound = new WeakSet();
    let sequence = 0, viewport = [], trialRows = [], highlighted = null, lastLocated = null;

    function same(a, b) { return JSON.stringify(a) === JSON.stringify(b); }
    function graph(name) { return document.querySelector("#" + prefix + name + " .js-plotly-plot"); }
    function rowId(target) {
        if (!target || !target.closest) return null;
        const cell = target.closest("#" + prefix + "table td[data-dash-row]");
        const row = cell && viewport[Number(cell.getAttribute("data-dash-row"))];
        return row ? row.id : null;
    }
    function paintRows() {
        document.querySelectorAll("#" + prefix + "table tr").forEach(function (row) {
            row.classList.toggle("sensitivity-hover", Boolean(highlighted && rowId(row.querySelector("td[data-dash-row]")) === highlighted));
        });
    }
    function highlight(id) {
        if (highlighted === id) return;
        highlighted = id;
        paintRows();
        ["scatter", "global-scatter"].forEach(function (name) {
            const plot = graph(name);
            if (!plot || !plot.data || !plot.data[0] || !window.Plotly) return;
            const indices = plot.data.flatMap(function (trace, i) { return trace.customdata ? [i] : []; });
            const opacity = indices.map(function (i) {
                return plot.data[i].customdata.map(function (item) { return !id || item[0] === id ? 1 : .3; });
            });
            window.Plotly.restyle(plot, {"marker.opacity": opacity}, indices);
        });
    }
    function emit(action, ids) {
        client.set_props(prefix + "plot-event", {data: {action: action, ids: ids, sequence: ++sequence}});
    }
    function bindPlots() {
        ["scatter", "global-scatter"].forEach(function (name) {
        const scatter = graph(name);
        if (scatter && scatter.on && !bound.has(scatter)) {
            bound.add(scatter);
            if (name === "global-scatter") {
                scatter.on("plotly_click", function (event) {
                    const point = event.points.find(function (p) { return p.customdata; });
                    if (point) emit("activate", [point.customdata[0]]);
                });
            }
            scatter.on("plotly_selected", function (event) {
                if (!event || !event.points.length) return;
                emit("compare", Array.from(new Set(event.points.filter(function (p) { return p.customdata; }).map(function (p) { return p.customdata[0]; }))));
                requestAnimationFrame(function () {
                    window.Plotly.relayout(scatter, {selections: []});
                    const indices = scatter.data.flatMap(function (trace, i) { return trace.customdata ? [i] : []; });
                    window.Plotly.restyle(scatter, {selectedpoints: [null]}, indices);
                });
            });
            scatter.on("plotly_hover", function (event) {
                const point = event.points.find(function (p) { return p.customdata; });
                if (point) highlight(point.customdata[0]);
            });
            scatter.on("plotly_unhover", function () { highlight(null); });
        }
        });
        const distribution = graph("distribution");
        if (distribution && distribution.on && !bound.has(distribution)) {
            bound.add(distribution);
            distribution.on("plotly_click", function (event) {
                const point = event.points.find(function (p) { return p.customdata; });
                if (!point) return;
                const id = point.customdata[0];
                const index = trialRows.findIndex(function (row) { return row.id === id; });
                if (index < 0 && point.customdata[1]) {
                    emit("activate", [point.customdata[1]]);
                    return;
                }
                client.set_props(prefix + "trials", {page_current: index < 0 ? 0 : Math.floor(index / 12), style_data_conditional: [
                    {if: {filter_query: '{id} = "' + id + '"'}, backgroundColor: "#eef2ff"},
                ]});
            });
        }
    }
    client.sensitivity = {
        trialRows: function (rows) {
            trialRows = rows || [];
            return client.no_update;
        },
        visibleRows: function (rows) {
            viewport = rows || [];
            requestAnimationFrame(paintRows);
            return client.no_update;
        },
        selection: function (cell, checked, table, visible, event, locate, active, comparison, data, candidates) {
            const changes = client.callback_context.triggered.map(function (item) { return item.prop_id; });
            const all = new Map((table || []).map(function (r) { return [r.id, r]; }));
            const ids = new Set(visible === null || visible === undefined ? Array.from(all.keys()) : visible);
            let nextActive = ids.has(active) ? active : Array.from(ids)[0] || null;
            let selected = (comparison || []).filter(function (id) { return ids.has(id); });
            if (changes.includes(prefix + "table.active_cell") && cell && cell.row_id && ids.has(cell.row_id)) {
                nextActive = cell.row_id;
            } else if (changes.includes(prefix + "table.selected_row_ids") &&
                    !changes.includes(prefix + "table.data") && !changes.includes(prefix + "table.derived_virtual_row_ids")) {
                selected = (checked || []).filter(function (id) { return ids.has(id); });
            } else if (changes.includes(prefix + "plot-event.data") && event) {
                if (event.action === "activate" && ids.has(event.ids[0])) nextActive = event.ids[0];
                if (event.action === "compare") selected = Array.from(new Set(selected.concat(event.ids))).filter(function (id) { return ids.has(id); });
            }
            const query = new URLSearchParams(window.location.search).get("trial");
            if (changes.includes(prefix + "locate.n_clicks") || (query && query !== lastLocated)) {
                const requested = query ? [query] : Object.keys(candidates || {}).reverse();
                const group = ((data || {}).groups || []).find(function (g) {
                    return ids.has(g.id) && g.rows.some(function (r) { return requested.includes(r.id); });
                });
                if (group) { nextActive = group.id; lastLocated = query; }
            }
            const styles = [{if: {column_id: "label"}, textAlign: "left"}];
            ((data || {}).groups || []).forEach(function (group) {
                if (ids.has(group.id)) styles.push({
                    if: {filter_query: '{id} = "' + group.id + '"', column_id: "label"},
                    borderLeft: "3px solid " + group.color,
                });
            });
            selected.forEach(function (id) { styles.push({if: {filter_query: '{id} = "' + id + '"'}, backgroundColor: "#f1f5f9"}); });
            if (nextActive) styles.push({if: {filter_query: '{id} = "' + nextActive + '"'}, backgroundColor: "#eef2ff", fontWeight: "600"});
            requestAnimationFrame(bindPlots);
            return [nextActive === active ? client.no_update : nextActive,
                same(selected, comparison || []) ? client.no_update : selected,
                same(selected, checked || []) ? client.no_update : selected,
                (table || []).flatMap(function (row, i) { return selected.includes(row.id) ? [i] : []; }),
                cell ? null : client.no_update, styles];
        },
        addCandidates: function (allClicks, checkedClicks, active, data, trials, saved) {
            const trigger = client.callback_context.triggered_id;
            if ((trigger === prefix + "add-config" && !allClicks) || (trigger === prefix + "add-trials" && !checkedClicks)) {
                return [client.no_update, client.no_update];
            }
            const group = ((data || {}).groups || []).find(function (g) { return g.id === active; });
            if (!group) return [client.no_update, "Activate a configuration first."];
            const rows = group.rows.filter(function (row) { return trigger === prefix + "add-config" || (trials || []).includes(row.id); });
            const next = Object.assign({}, saved || {});
            rows.forEach(function (row) { if (!next[row.id]) next[row.id] = row; });
            return [next, rows.length + " trials added to Select Candidates · " + Object.keys(next).length + " trials in the shared collection."];
        },
    };
    document.addEventListener("mouseover", function (event) { const id = rowId(event.target); if (id) highlight(id); });
    document.addEventListener("mouseout", function (event) { if (rowId(event.target) && !rowId(event.relatedTarget)) highlight(null); });
    function observe() {
        new MutationObserver(function () {
            if (document.getElementById(prefix + "workspace")) bindPlots();
        }).observe(document.body, {childList: true, subtree: true});
    }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", observe);
    else observe();
})();
