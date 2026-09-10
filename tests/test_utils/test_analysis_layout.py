import plotly.graph_objects as go
import pytest

from deepcave.utils.analysis_layout import bin_label, bin_scale_exponent, style_analysis


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # pandas extends the lowest bin edge slightly; the label shows the bin's own precision.
        ("(0.499, 0.55]", "0.50\u2060–\u20600.55"),
        ("(0.55, 0.6]", "0.55\u2060–\u20600.60"),
        ("(0.9, 1.0]", "0.9\u2060–\u20601.0"),
        ("(0.09999999999999998, 0.2]", "0.1\u2060–\u20600.2"),
        ("(4.995, 15.0]", "5\u2060–\u206015"),
        ("(1e-05, 3.98e-05]", "1e-05\u2060–\u20604e-05"),
        ("(0.004, 0.007]", "0.004\u2060–\u20600.007"),
        ("a", "a"),
        ("(low, high]", "(low, high]"),
    ],
)
def test_bin_labels_are_rounded_to_the_precision_of_the_bin(value, expected):
    assert bin_label(value, multiline=False) == expected


def test_bin_labels_break_into_two_lines_by_default():
    assert bin_label("(0.499, 0.55]") == "0.50 –<br>0.55"


def test_small_bin_labels_share_a_scale_factor_for_the_axis_title():
    ranges = ["(0.004, 0.007]", "(0.007, 0.009]", "(0.009, 0.011]"]
    exponent = bin_scale_exponent(ranges)

    assert exponent == -3
    assert [
        bin_label(value, multiline=False, scale_exponent=exponent) for value in ranges
    ] == ["4\u2060–\u20607", "7\u2060–\u20609", "9\u2060–\u206011"]


def test_hiding_the_legend_frees_the_reserved_bottom_space():
    with_legend, without_legend = go.Figure(), go.Figure()

    style_analysis(with_legend)
    style_analysis(without_legend, show_legend=False)

    assert with_legend.layout.margin.b > without_legend.layout.margin.b
    assert without_legend.layout.showlegend is False
