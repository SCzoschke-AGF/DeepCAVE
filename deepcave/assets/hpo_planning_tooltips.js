(function () {
    "use strict";

    window.dccFunctions = window.dccFunctions || {};

    const transforms = {};
    const numberFormatter = new Intl.NumberFormat(undefined, {
        maximumSignificantDigits: 6,
        useGrouping: false,
    });

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
})();
