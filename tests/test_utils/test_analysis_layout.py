import plotly.graph_objects as go
import pytest

from deepcave.utils.analysis_layout import bin_label, style_analysis


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # pandas extends the lowest bin edge slightly; the label shows the bin's own precision.
        ("(0.499, 0.55]", "0.50 – 0.55"),
        ("(0.55, 0.6]", "0.55 – 0.60"),
        ("(0.9, 1.0]", "0.9 – 1.0"),
        ("(0.09999999999999998, 0.2]", "0.1 – 0.2"),
        ("(4.995, 15.0]", "5 – 15"),
        ("(1e-05, 3.98e-05]", "1e-05 – 4e-05"),
        ("a", "a"),
        ("(low, high]", "(low, high]"),
    ],
)
def test_bin_labels_are_rounded_to_the_precision_of_the_bin(value, expected):
    assert bin_label(value, multiline=False) == expected


def test_bin_labels_break_into_two_lines_by_default():
    assert bin_label("(0.499, 0.55]") == "0.50 –<br>0.55"


def test_hiding_the_legend_frees_the_reserved_bottom_space():
    with_legend, without_legend = go.Figure(), go.Figure()

    style_analysis(with_legend)
    style_analysis(without_legend, show_legend=False)

    assert with_legend.layout.margin.b > without_legend.layout.margin.b
    assert without_legend.layout.showlegend is False
