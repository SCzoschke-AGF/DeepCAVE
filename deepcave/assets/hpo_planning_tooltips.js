(function () {
    "use strict";

    window.dccFunctions = window.dccFunctions || {};

    const transforms = {};
    const observedTracks = new WeakSet();
    const pendingSliders = new WeakSet();
    const numberFormatter = new Intl.NumberFormat(undefined, {
        maximumSignificantDigits: 6,
        useGrouping: false,
    });

    function updateUnselectedRegions(slider) {
        const control = slider.closest(".planner-range-control");
        const rail = slider.querySelector(".rc-slider-rail");
        const track = slider.querySelector(".rc-slider-track");
        const regions = control && control.querySelectorAll(
            ".planner-unselected-layer .planner-unselected-segment"
        );
        if (!control || !rail || !track || !regions || regions.length !== 2) {
            return;
        }

        const railBounds = rail.getBoundingClientRect();
        const trackBounds = track.getBoundingClientRect();
        if (railBounds.width <= 0) {
            return;
        }

        const selectedStart = Math.min(
            100,
            Math.max(0, (100 * (trackBounds.left - railBounds.left)) / railBounds.width)
        );
        const selectedEnd = Math.min(
            100,
            Math.max(0, (100 * (trackBounds.right - railBounds.left)) / railBounds.width)
        );
        regions[0].style.left = "0";
        regions[0].style.width = `${selectedStart}%`;
        regions[1].style.left = `${selectedEnd}%`;
        regions[1].style.width = `${100 - selectedEnd}%`;
    }

    function scheduleUnselectedUpdate(slider) {
        if (pendingSliders.has(slider)) {
            return;
        }
        pendingSliders.add(slider);
        window.requestAnimationFrame(function () {
            pendingSliders.delete(slider);
            updateUnselectedRegions(slider);
        });
    }

    function observeSlider(slider) {
        const track = slider.querySelector(".rc-slider-track");
        if (!track || observedTracks.has(track)) {
            return;
        }
        observedTracks.add(track);
        new MutationObserver(function () {
            scheduleUnselectedUpdate(slider);
        }).observe(track, {attributes: true, attributeFilter: ["style"]});
        scheduleUnselectedUpdate(slider);
    }

    function observePlannerSliders() {
        document.querySelectorAll(".planner-range-slider").forEach(observeSlider);
    }

    function findSlider(index) {
        const sliders = document.querySelectorAll(".planner-range-slider");
        for (const slider of sliders) {
            try {
                if (JSON.parse(slider.id).index === index) {
                    return slider;
                }
            } catch (_error) {
                // Ignore unrelated elements whose IDs are not Dash pattern IDs.
            }
        }
        return null;
    }

    function originalValue(index, position) {
        const slider = findSlider(index);
        const control = slider && slider.closest(".planner-range-control");
        if (!control) {
            return position;
        }

        const lower = Number(control.dataset.currentMin);
        const upper = Number(control.dataset.currentMax);
        const logarithmic = control.dataset.log === "true";
        let value;
        if (logarithmic) {
            value = Math.exp(
                Math.log(lower) + Number(position) * (Math.log(upper) - Math.log(lower))
            );
        } else {
            value = lower + Number(position) * (upper - lower);
        }

        const step = Number(control.dataset.step);
        if (control.dataset.step !== "" && step > 0) {
            value = lower + Math.round((value - lower) / step) * step;
        }
        if (control.dataset.parameterType === "integer") {
            value = Math.round(value);
        }
        return numberFormatter.format(value);
    }

    window.dccFunctions.hpoPlanning = new Proxy(transforms, {
        get: function (target, property) {
            if (!(property in target)) {
                const match = /^parameter(\d+)$/.exec(String(property));
                if (!match) {
                    return undefined;
                }
                const index = Number(match[1]);
                target[property] = function (position) {
                    return originalValue(index, position);
                };
            }
            return target[property];
        },
    });

    // Dash replaces planner rows after server callbacks. Observe both the current
    // sliders and any replacements so the gray regions follow the native track
    // immediately while dragging, independently of callback latency.
    observePlannerSliders();
    new MutationObserver(observePlannerSliders).observe(document.documentElement, {
        childList: true,
        subtree: true,
    });
    window.addEventListener("resize", function () {
        document.querySelectorAll(".planner-range-slider").forEach(scheduleUnselectedUpdate);
    });
})();
