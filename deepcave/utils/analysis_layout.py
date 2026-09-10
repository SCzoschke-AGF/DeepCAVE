"""Consistent, readable layout for empirical analysis views and figures."""

from typing import Any, Dict, List, Optional

import math
import re

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html
from dash.development.base_component import Component

from deepcave.utils.layout import help_button

PANEL_LEFT_MARGIN = 64
PANEL_TOP_MARGIN = 44
COLORBAR_CONTAINER_Y = 0.015
COLORBAR_THICKNESS = 10
COLORBAR_TICK_HEIGHT = 24
COLORBAR_LABEL_PADDING = 7
NONBREAKING_RANGE_SEPARATOR = "\u2060–\u2060"
SUPERSCRIPT_TRANSLATION = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def analysis_toolbar_item(
    label: str,
    control: Component,
    width: str,
    help_text: Optional[str] = None,
) -> Component:
    """Wrap one control in the compact toolbar style shared by analysis plugins."""
    children: List[Component] = [
        dbc.InputGroup(
            [dbc.InputGroupText(label), control],
            size="sm",
            className="flex-nowrap",
        )
    ]
    if help_text is not None:
        children.append(help_button(help_text))

    return html.Div(
        children,
        className="analysis-toolbar__item d-flex align-items-center me-2",
        style={"flex": f"1 1 {width}", "minWidth": width},
    )


def analysis_graph(graph_id: str, height: int, download_scale: float) -> Component:
    """Return a responsive graph sized for an analysis card."""
    return dcc.Graph(
        id=graph_id,
        responsive=True,
        style={"width": "100%", "height": f"{height}px"},
        config={"toImageButtonOptions": {"scale": download_scale}},
    )


def analysis_card(title: str, graph: Component, modifier: str = "") -> Component:
    """Put a graph below an external card header, keeping Plotly titles out of the canvas."""
    class_name = "analysis-card"
    if modifier:
        class_name += f" {modifier}"
    return html.Div(
        [html.Div(title, className="analysis-card__header"), graph],
        className=class_name,
    )


def style_analysis_panel(
    figure: go.Figure,
    height: int,
    bottom: int,
    show_legend: bool = False,
    top: int = PANEL_TOP_MARGIN,
    left: int = PANEL_LEFT_MARGIN,
    right: int = 22,
) -> None:
    """Apply the plot styling used inside analysis cards."""
    style_analysis(figure, height=height, show_legend=show_legend)
    figure.update_layout(
        title=None,
        margin={"t": top, "b": bottom, "l": left, "r": right},
        legend={
            "orientation": "h",
            "x": 1,
            "xanchor": "right",
            "y": 1.01,
            "yanchor": "bottom",
            "font": {"size": 12},
        },
    )
    figure.update_yaxes(automargin=False)
    figure.update_xaxes(automargin=False)


def horizontal_colorbar() -> Dict[str, Any]:
    """Return a compact colorbar positioned below an analysis plot."""
    return {
        "orientation": "h",
        "x": 0.5,
        "xanchor": "center",
        "yref": "container",
        "y": COLORBAR_CONTAINER_Y,
        "yanchor": "bottom",
        "thickness": COLORBAR_THICKNESS,
        "len": 0.8,
        "tickfont": {"size": 10},
        "nticks": 6,
    }


def add_colorbar_label(
    figure: go.Figure,
    text: str,
    height: int,
    bottom: int,
    top: int = PANEL_TOP_MARGIN,
) -> None:
    """Place a centered caption immediately above a horizontal colorbar."""
    above_figure_bottom = (
        COLORBAR_CONTAINER_Y * height
        + COLORBAR_TICK_HEIGHT
        + COLORBAR_THICKNESS
        + COLORBAR_LABEL_PADDING
    )
    figure.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=0.5,
        xanchor="center",
        y=-(bottom - above_figure_bottom) / (height - top - bottom),
        yanchor="bottom",
        showarrow=False,
        font={"size": 11},
    )


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


def bin_scale_exponent(values: List[str]) -> Optional[int]:
    """Return a shared engineering exponent for a sequence of small numeric intervals."""
    bounds = []
    for value in values:
        match = re.fullmatch(r"[\[(]([^,]+),\s*([^\]]+)\]", value)
        if not match:
            return None
        try:
            bounds.extend([float(match[1]), float(match[2])])
        except ValueError:
            return None

    maximum_magnitude = max((abs(bound) for bound in bounds), default=0.0)
    if 0 < maximum_magnitude < 0.1:
        return 3 * math.floor(math.log10(maximum_magnitude) / 3)
    return None


def superscript_exponent(exponent: int) -> str:
    """Format a signed integer exponent with Unicode superscript characters."""
    return str(exponent).translate(SUPERSCRIPT_TRANSLATION)


def bin_label(
    value: str,
    multiline: bool = True,
    scale_exponent: Optional[int] = None,
) -> str:
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

    if scale_exponent is not None:
        scale = 10**scale_exponent
        lower /= scale
        upper /= scale

    width = upper - lower
    separator = " –<br>" if multiline else NONBREAKING_RANGE_SEPARATOR
    return f"{_bound_label(lower, width)}{separator}{_bound_label(upper, width)}"
