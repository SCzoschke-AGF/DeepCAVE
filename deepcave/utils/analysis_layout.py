"""Consistent, readable layout for empirical analysis figures."""

import math
import re

import plotly.graph_objects as go


def style_analysis(figure: go.Figure, height: int = 500, show_legend: bool = True) -> None:
    """
    Reserve separate space for axes, titles, and the shared bottom legend.

    Parameters
    ----------
    figure : go.Figure
        The figure to style in place.
    height : int, optional
        Figure height in pixels, by default 500.
    show_legend : bool, optional
        Whether a shared bottom legend is rendered. Dense single-screen views encode their
        series in the subplot titles instead and therefore do not reserve bottom space.
    """
    figure.update_layout(
        template="plotly_white",
        height=height,
        autosize=True,
        font=dict(family="Arial, sans-serif", size=13, color="#334155"),
        title=dict(font=dict(size=16), x=0.02),
        margin=dict(t=65, b=130 if show_legend else 60, l=75, r=45),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            x=0,
            xanchor="left",
            y=-0.32,
            yanchor="top",
            font=dict(size=12),
            groupclick="togglegroup",
        ),
        hoverlabel=dict(font_size=12),
    )
    figure.update_xaxes(
        automargin=True,
        tickfont_size=12,
        title_font_size=13,
        title_standoff=14,
        gridcolor="#edf0f4",
        zeroline=False,
    )
    figure.update_yaxes(
        automargin=True,
        tickfont_size=12,
        title_font_size=13,
        title_standoff=12,
        gridcolor="#edf0f4",
        zeroline=False,
    )
    figure.update_annotations(font_size=14)


def _bound_label(value: float, width: float) -> str:
    """Format a single interval bound with a precision matching the bin width."""
    if width <= 0 or not math.isfinite(width):
        return f"{value:.3g}"

    magnitude = abs(value)
    if magnitude != 0 and (magnitude < 1e-3 or magnitude >= 1e5):
        return f"{value:.2g}"

    # Rounding the exponent first keeps a series of bins consistent when a width such as
    # 0.1 is stored as 0.09999999999999998.
    decimals = max(0, min(6, math.ceil(round(-math.log10(width), 6))))
    return f"{value:.{decimals}f}"


def bin_label(value: str, multiline: bool = True) -> str:
    """
    Display interval bounds on two compact lines; preserve full values in hover.

    Bounds are rounded to the precision the bin width actually carries, so a bin that
    pandas reports as ``(0.499, 0.55]`` reads as ``0.50 – 0.55``.
    """
    match = re.fullmatch(r"[\[(]([^,]+),\s*([^\]]+)\]", value)
    if not match:
        return value

    try:
        lower, upper = float(match[1]), float(match[2])
    except ValueError:
        return value

    width = upper - lower
    separator = " –<br>" if multiline else " – "
    return f"{_bound_label(lower, width)}{separator}{_bound_label(upper, width)}"
