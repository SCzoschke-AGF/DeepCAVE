/* One session selection reducer; hover changes presentation without server round trips. */
(function () {
    "use strict";
    const client = window.dash_clientside = window.dash_clientside || {};
    const prefix = "select-candidates-";
    let selection = {};
    let highlighted = null;
    let sequence = 0;
    let tableRows = [];
    const bound = new WeakSet();
    const scheduled = new WeakSet();

    function schedulePaint(plot, painter) {
        if (!plot || scheduled.has(plot)) return;
        scheduled.add(plot);
        requestAnimationFrame(function () { scheduled.delete(plot); painter(plot); });
    }

    function graph(name) {
        return document.querySelector("#" + prefix + name + " .js-plotly-plot");
    }

    function emit(action, ids) {
        client.set_props(prefix + "event", {data: {action: action, ids: ids, sequence: ++sequence}});
    }

    function same(left, right) {
        return JSON.stringify(left) === JSON.stringify(right);
    }

    function paintScatter(plot) {
        if (!window.Plotly || !plot || !plot.data || !plot.data[0] ||
                !plot.layout.meta || !plot.layout.meta.candidate_scatter) return;
        const trace = plot.data[0];
        const colors = [], sizes = [], opacity = [];
        (trace.customdata || []).forEach(function (item) {
            const selected = Boolean(selection[item[0]]);
            colors.push(selected ? selection[item[0]].color : "#a6adbb");
            sizes.push(item[0] === highlighted ? 13 : selected ? 9 : 6);
            opacity.push(highlighted && item[0] !== highlighted ? 0.3 : selected ? 1 : 0.65);
        });
        if (!same(colors, trace.marker.color) || !same(sizes, trace.marker.size) ||
                !same(opacity, trace.marker.opacity)) {
            window.Plotly.restyle(plot, {"marker.color": [colors], "marker.size": [sizes],
                "marker.opacity": [opacity]}, [0]);
        }
    }

    function paintParallel(plot) {
        if (!window.Plotly || !plot || !plot.data || plot.data.length < 2 ||
                !plot.layout.meta || !plot.layout.meta.candidate_parallel) return;
        const ids = plot.data[0].meta.candidate_ids;
        const index = ids.indexOf(highlighted);
        const overlay = plot.data[1];
        if (overlay.meta && overlay.meta.highlighted === highlighted) return;
        const dimensions = plot.data[0].dimensions.map(function (dimension) {
            const updated = Object.assign({}, dimension, {
                values: index < 0 ? [] : [dimension.values[index]],
            });
            delete updated.constraintrange;
            return updated;
        });
        window.Plotly.restyle(plot, {dimensions: [dimensions],
            "line.color": index < 0 ? "#172554" : selection[highlighted].color,
            meta: {highlighted: highlighted}}, [1]);
    }

    function rowKey(element) {
        if (!element || !element.closest) return null;
        const chip = element.closest("[data-candidate-key]");
        if (chip) return chip.dataset.candidateKey;
        const cell = element.closest("#" + prefix + "table td[data-dash-row]");
        if (!cell) return null;
        const index = Number(cell.getAttribute("data-dash-row"));
        const row = tableRows[index];
        return row ? row.id : null;
    }

    function paintRows() {
        document.querySelectorAll("#" + prefix + "table tr, .candidate-chip").forEach(function (element) {
            const cell = element.matches("tr") ? element.querySelector("td[data-dash-row]") : element;
            element.classList.toggle("candidate-highlight", Boolean(highlighted && rowKey(cell) === highlighted));
        });
    }

    function highlight(key) {
        if (key === highlighted) return;
        highlighted = key;
        client.set_props(prefix + "highlighted", {data: key});
        schedulePaint(graph("scatter"), paintScatter);
        schedulePaint(graph("parallel"), paintParallel);
        paintRows();
    }

    client.candidateSelection = {
        reduce: function (checked, active, event, removed, clear, table, data, saved) {
            const context = client.callback_context;
            const trigger = context.triggered_id;
            const changes = context.triggered.map(function (item) { return item.prop_id; });
            const unchanged = client.no_update;
            const next = Object.assign({}, saved || {});
            const available = new Map(((data || {}).rows || []).map(function (row) { return [row.id, row]; }));
            let mutated = false;
            function add(id) {
                if (next[id] || !available.has(id)) return;
                const row = available.get(id);
                const used = new Set(Object.values(next).map(function (item) { return item.color; }));
                const color = used.has(row.color)
                    ? ((data.palette || []).find(function (value) { return !used.has(value); }) || row.color)
                    : row.color;
                next[id] = Object.assign({}, row, {color: color});
                mutated = true;
            }
            function toggle(id) {
                if (next[id]) { delete next[id]; mutated = true; }
                else add(id);
            }
            if (trigger === prefix + "clear" && clear > 0) {
                Object.keys(next).forEach(function (key) { delete next[key]; });
                mutated = true;
            } else if (trigger && trigger.type === "candidate-remove" &&
                    (removed || []).some(function (value) { return value > 0; })) {
                delete next[trigger.key];
                mutated = true;
            } else if (trigger === prefix + "event" && event) {
                if (event.action === "toggle") (event.ids || []).forEach(toggle);
                else (event.ids || []).forEach(add);
            } else if (changes.includes(prefix + "table.active_cell") && active && active.row_id) {
                toggle(active.row_id);
            } else if (changes.includes(prefix + "table.selected_row_ids") &&
                    !changes.includes(prefix + "table.data") && !changes.includes(prefix + "data.data")) {
                const ids = new Set(checked || []);
                (table || []).forEach(function (row) {
                    if (ids.has(row.id) && !next[row.id] && available.has(row.id)) {
                        add(row.id);
                    } else if (!ids.has(row.id) && next[row.id]) {
                        delete next[row.id]; mutated = true;
                    }
                });
            }
            selection = next;
            const ids = (table || []).filter(function (row) { return Boolean(next[row.id]); }).map(function (row) { return row.id; });
            const indices = (table || []).flatMap(function (row, index) { return next[row.id] ? [index] : []; });
            const styles = [{if: {column_id: "study_name"}, textAlign: "left"}];
            ids.forEach(function (id) {
                styles.push({if: {filter_query: '{id} = "' + id + '"'}, backgroundColor: "#eef2ff"});
            });
            requestAnimationFrame(function () {
                bindPlots(); paintScatter(graph("scatter")); paintParallel(graph("parallel")); paintRows();
            });
            return [mutated ? next : unchanged, same(ids, checked || []) ? unchanged : ids,
                indices, active ? null : unchanged, styles];
        },
        visibleRows: function (rows, table) {
            tableRows = rows || table || [];
            requestAnimationFrame(paintRows);
            return client.no_update;
        },
    };

    function bindPlots() {
        const scatter = graph("scatter");
        if (scatter && scatter.on && !bound.has(scatter)) {
            bound.add(scatter);
            scatter.on("plotly_click", function (event) {
                const point = event.points.find(function (item) { return item.customdata; });
                if (point) emit("toggle", [point.customdata[0]]);
            });
            scatter.on("plotly_selected", function (event) {
                if (event && event.points.length) emit("add", Array.from(new Set(event.points.filter(function (point) {
                    return point.customdata;
                }).map(function (point) { return point.customdata[0]; }))));
                // Collected trials own their highlight. Remove Plotly's movable gesture outline
                // so the next additive lasso can start inside the previous box.
                if (event && event.points.length) requestAnimationFrame(function () {
                    window.Plotly.relayout(scatter, {selections: []});
                    window.Plotly.restyle(scatter, {selectedpoints: [null]}, [0]);
                });
            });
            scatter.on("plotly_hover", function (event) {
                const point = event.points.find(function (item) { return item.customdata; });
                if (point) highlight(point.customdata[0]);
            });
            scatter.on("plotly_unhover", function () { highlight(null); });
            scatter.on("plotly_afterplot", function () { schedulePaint(scatter, paintScatter); });
            paintScatter(scatter);
        }
        const parallel = graph("parallel");
        if (parallel && parallel.on && !bound.has(parallel)) {
            bound.add(parallel);
            parallel.on("plotly_afterplot", function () { schedulePaint(parallel, paintParallel); });
            // Axis order/brush is a comparison view operation, never a permanent selection.
            parallel.on("plotly_restyle", function (event) {
                if (event && event[0] && Object.keys(event[0]).some(function (key) {
                    return key === "dimensions" || key.includes("constraintrange");
                }) && (!event[1] || event[1].includes(0))) {
                    parallel.data[1].meta = null;
                    schedulePaint(parallel, paintParallel);
                }
            });
            parallel.addEventListener("pointermove", function (event) {
                if (event.buttons || event.target.closest(".axis-heading, .axis-brush")) return;
                const trace = parallel.data[0];
                if (!trace || !trace.dimensions) return;
                // Parcoords has no native line hover. Hit-test its rendered axis geometry.
                const axes = Array.from(parallel.querySelectorAll(".parcoords .y-axis"))
                    .slice(0, trace.dimensions.length);
                if (axes.length !== trace.dimensions.length) return;
                const geometry = axes.map(function (axis) {
                    const path = axis.querySelector(".axis .domain");
                    return path ? path.getBoundingClientRect() : null;
                });
                if (geometry.some(function (item) { return !item; })) return;
                geometry.sort(function (left, right) { return left.x - right.x; });
                const x = event.clientX, y = event.clientY;
                const segment = geometry.findIndex(function (rect, i) {
                    return i + 1 < geometry.length && x >= rect.x && x <= geometry[i + 1].x;
                });
                if (segment < 0) return;
                let best = 8, match = null;
                trace.meta.candidate_ids.forEach(function (id, rowIndex) {
                    function ordinate(index) {
                        const dimension = trace.dimensions[index];
                        const range = dimension.range;
                        const rect = geometry[index];
                        return rect.bottom - (dimension.values[rowIndex] - range[0]) /
                            (range[1] - range[0]) * rect.height;
                    }
                    const fraction = (x - geometry[segment].x) / (geometry[segment + 1].x - geometry[segment].x);
                    const distance = Math.abs(y - (ordinate(segment) * (1 - fraction) + ordinate(segment + 1) * fraction));
                    if (distance < best) { best = distance; match = id; }
                });
                highlight(match);
                const rows = parallel.layout.meta.rows;
                const row = rows.find(function (item) { return item.id === match; });
                showTooltip(row, x, y);
            });
            parallel.addEventListener("pointerleave", function () { highlight(null); showTooltip(null); });
            paintParallel(parallel);
        }
    }

    function showTooltip(row, x, y) {
        let tooltip = document.querySelector(".candidate-parallel-tooltip");
        if (!row) { if (tooltip) tooltip.remove(); return; }
        if (!tooltip) {
            tooltip = document.createElement("div");
            tooltip.className = "candidate-parallel-tooltip";
            document.body.appendChild(tooltip);
        }
        tooltip.textContent = row.study_name + " · Trial " + row.trial_number;
        tooltip.style.left = (x + 12) + "px";
        tooltip.style.top = (y + 12) + "px";
    }

    ["mouseover", "focusin"].forEach(function (name) {
        document.addEventListener(name, function (event) {
            const key = rowKey(event.target);
            if (key) highlight(key);
        });
    });
    ["mouseout", "focusout"].forEach(function (name) {
        document.addEventListener(name, function (event) {
            if (rowKey(event.target) && !rowKey(event.relatedTarget)) highlight(null);
        });
    });
    const observer = new MutationObserver(function () {
        if (document.getElementById(prefix + "workspace")) bindPlots();
    });
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", function () { observer.observe(document.body, {childList: true, subtree: true}); });
    } else observer.observe(document.body, {childList: true, subtree: true});
})();
