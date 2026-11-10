/* Native disclosure menus; the existing Dash checklist retains all selected values. */
(function () {
    "use strict";
    const client = window.dash_clientside = window.dash_clientside || {};
    const optionsById = {};

    function filter(menu) {
        const query = menu.querySelector(".candidate-selector-search").value.trim().toLocaleLowerCase();
        const labels = menu.querySelectorAll(".candidate-selector-options label");
        let visible = 0;
        labels.forEach(function (label) {
            label.hidden = !label.textContent.toLocaleLowerCase().includes(query);
            if (!label.hidden) visible++;
        });
        menu.querySelector(".candidate-selector-empty").hidden = visible > 0;
    }

    client.candidateSelectors = {
        summarize: function (options, values, id) {
            options = options || [];
            values = values || [];
            optionsById[id] = options;
            const noun = id.endsWith("objectives") ? "objectives" : "studies";
            const placeholder = noun === "objectives" ? "Select objectives …" : "Add loaded studies …";
            const selected = options.filter(option => values.includes(option.value));
            const text = !selected.length ? placeholder : selected.length === 1 ?
                selected[0].label : selected.length + " " + noun + " selected";
            // Apply the current query after Dash has rendered refreshed options.
            requestAnimationFrame(function () {
                const list = document.getElementById(id);
                if (list) filter(list.closest(".candidate-selector"));
            });
            return [text, text, selected.length ? "" : "candidate-selector-placeholder",
                selected.length + " selected"];
        }
    };

    document.addEventListener("input", function (event) {
        if (event.target.matches(".candidate-selector-search")) {
            filter(event.target.closest(".candidate-selector"));
        }
    });
    document.addEventListener("click", function (event) {
        document.querySelectorAll(".candidate-selector[open]").forEach(function (menu) {
            if (!menu.contains(event.target)) menu.open = false;
        });
        const button = event.target.closest(".candidate-selector-actions button");
        if (!button) return;
        const list = button.closest(".candidate-selector").querySelector(".candidate-selector-options");
        client.set_props(list.id, {value: button.dataset.action === "all" ?
            (optionsById[list.id] || []).map(option => option.value) : []});
    });
    document.addEventListener("keydown", function (event) {
        const menu = event.target.closest(".candidate-selector[open]");
        if (event.key === "Escape" && menu) {
            menu.open = false;
            menu.querySelector("summary").focus();
        }
    });
}());
