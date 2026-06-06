from babygrad import formatting


def test_vector_limits_float_decimal_places():
    output = formatting.vector([1.234567, 2.0])

    assert "1.2346" in output
    assert "1.234567" not in output
    assert "2.0" in output


def test_matrix_distinguishes_text_ellipsis_from_truncation_marker():
    output = formatting.matrix([1, "...", 3], nrow=1, ncol=3)

    assert "'...'" in output
