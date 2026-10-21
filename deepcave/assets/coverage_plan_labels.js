(function () {
    "use strict";

    // Keep labels near their actual segments, using extra rows only for collisions.
    function positionLabels(legend) {
        const width = legend.clientWidth;
        if (!width) return;
        legend.dataset.positioned = "true";
        const lanes = [];
        const heights = [];
        Array.from(legend.children).forEach(function (label) {
            label.style.maxWidth = width + "px";
            const bounds = label.getBoundingClientRect();
            const anchor = width * Number(label.dataset.anchor) / 100;
            const left = Math.max(0, Math.min(width - bounds.width, anchor - bounds.width / 2));
            let lane = 0;
            while (lanes[lane] && lanes[lane].some(function (entry) {
                return left < entry.right + 8 && left + bounds.width + 8 > entry.left;
            })) lane += 1;
            if (!lanes[lane]) lanes[lane] = [];
            lanes[lane].push({ label: label, left: left, right: left + bounds.width });
            heights[lane] = Math.max(heights[lane] || 0, bounds.height);
        });
        let bottom = 0;
        lanes.forEach(function (lane, index) {
            lane.forEach(function (entry) {
                entry.label.style.left = entry.left + "px";
                entry.label.style.bottom = bottom + "px";
            });
            bottom += heights[index] + 5;
        });
        legend.style.height = Math.max(0, bottom - 5) + "px";
    }

    function start() {
        const observed = new Set();
        let pending = false;
        function schedule() {
            if (pending) return;
            pending = true;
            window.requestAnimationFrame(function () {
                pending = false;
                observed.forEach(function (legend) {
                    if (!legend.isConnected) {
                        resize.unobserve(legend);
                        observed.delete(legend);
                    }
                });
                document.querySelectorAll(".coverage-plan-legend").forEach(function (legend) {
                    if (!observed.has(legend)) {
                        observed.add(legend);
                        resize.observe(legend);
                    }
                    positionLabels(legend);
                });
            });
        }
        const resize = new ResizeObserver(schedule);
        new MutationObserver(schedule).observe(document.body, {
            childList: true, subtree: true, characterData: true,
        });
        if (document.fonts) document.fonts.ready.then(schedule);
        schedule();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", start, { once: true });
    } else {
        start();
    }
}());
