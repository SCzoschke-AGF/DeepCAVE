/* Close the columns menu when clicking outside it, including after page navigation. */
(function () {
    "use strict";

    const widths = new Map();
    let drag = null;
    let frame = null;
    let suppressClickUntil = 0;

    function headers(table) {
        const rows = Array.from(table.querySelectorAll("tr"))
            .filter(function (row) { return row.querySelector("th.dash-header"); });
        return rows.length ? Array.from(rows[rows.length - 1].querySelectorAll("th.dash-header")) : [];
    }

    function applyWidths(table) {
        const columns = headers(table);
        let total = 0;
        columns.forEach(function (header) {
            const id = header.getAttribute("data-dash-column");
            const width = widths.get(id);
            total += width || header.getBoundingClientRect().width;
            if (!width) return;
            table.querySelectorAll("[data-dash-column]").forEach(function (cell) {
                if (cell.getAttribute("data-dash-column") !== id || cell.colSpan > 1) return;
                ["width", "min-width", "max-width"].forEach(function (property) {
                    cell.style.setProperty(property, width + "px", "important");
                });
                cell.classList.add("all-runs-width-set");
            });
        });
        if (columns.some(function (header) { return widths.has(header.getAttribute("data-dash-column")); })) {
            table.querySelectorAll("table").forEach(function (element) {
                element.style.width = total + "px";
                element.style.tableLayout = "auto";
            });
        }
    }

    function bindHeaders() {
        frame = null;
        const table = document.getElementById("all-runs-table");
        if (!table) return;
        headers(table).forEach(function (header) {
            header.classList.add("all-runs-resizable-header");
            if (header.querySelector(".all-runs-resize-handle")) return;
            const handle = document.createElement("span");
            handle.className = "all-runs-resize-handle";
            handle.title = "Drag to resize column";
            header.appendChild(handle);
        });
        applyWidths(table);
    }

    document.addEventListener("pointerdown", function (event) {
        document.querySelectorAll(".all-runs-column-menu[open]").forEach(function (menu) {
            if (!menu.contains(event.target)) {
                menu.open = false;
            }
        });
        const handle = event.target.closest(".all-runs-resize-handle");
        if (!handle || event.button !== 0) return;
        event.preventDefault();
        event.stopPropagation();
        const table = document.getElementById("all-runs-table");
        const header = handle.closest("th");
        headers(table).forEach(function (cell) {
            widths.set(cell.getAttribute("data-dash-column"), cell.getBoundingClientRect().width);
        });
        drag = {
            id: header.getAttribute("data-dash-column"),
            startX: event.clientX,
            startWidth: header.getBoundingClientRect().width,
        };
        document.body.classList.add("all-runs-resizing");
        applyWidths(table);
    }, true);

    document.addEventListener("pointermove", function (event) {
        if (!drag) return;
        event.preventDefault();
        widths.set(drag.id, Math.max(72, Math.round(drag.startWidth + event.clientX - drag.startX)));
        const table = document.getElementById("all-runs-table");
        if (table) applyWidths(table);
    }, true);

    function finishDrag() {
        if (!drag) return;
        drag = null;
        suppressClickUntil = Date.now() + 250;
        document.body.classList.remove("all-runs-resizing");
    }

    document.addEventListener("pointerup", finishDrag, true);
    document.addEventListener("pointercancel", finishDrag, true);
    window.addEventListener("blur", finishDrag);
    document.addEventListener("click", function (event) {
        if (event.target.closest(".all-runs-resize-handle") ||
            (Date.now() < suppressClickUntil && event.target.closest("#all-runs-table"))) {
            event.preventDefault();
            event.stopPropagation();
        }
    }, true);

    new MutationObserver(function () {
        if (frame === null) frame = window.requestAnimationFrame(bindHeaders);
    }).observe(document.documentElement, {childList: true, subtree: true});
    bindHeaders();
}());
