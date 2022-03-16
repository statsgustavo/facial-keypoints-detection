import hypothesis as hp
from hypothesis import strategies as st
from src import helpers


def test_load_metadata():
    metadata = helpers.load_metadata(["data"], ["raw"])
    assert isinstance(metadata, dict)
    assert "data" in metadata
    assert "training" in metadata["data"]
    assert "test" in metadata["data"]
    assert "delimiter" in metadata["data"]


def test_convolution_output_dimension():
    assert (4, 4, 5) == helpers.convolution_output_dimension(
        (6, 6), (3, 3), (1, 1), (0, 0), 5
    )
    assert (2, 2, 5) == helpers.convolution_output_dimension(
        (6, 6), (3, 3), (2, 2), (0, 0), 5
    )
    assert (6, 6, 5) == helpers.convolution_output_dimension(
        (6, 6), (3, 3), (1, 1), (1, 1), 5
    )
    assert (3, 3, 5) == helpers.convolution_output_dimension(
        (6, 6), (3, 3), (2, 2), (1, 1), 5
    )
    assert (3, 1, 5) == helpers.convolution_output_dimension(
        (6, 2), (3, 3), (2, 2), (1, 1), 5
    )
    assert (3, 1, 5) == helpers.convolution_output_dimension(
        (6, 3), (3, 4), (2, 2), (1, 1), 5
    )
    assert (10, 4, 15) == helpers.convolution_output_dimension(
        (12, 5), (3, 2), (1, 1), (0, 0), 15
    )


@hp.given(st.tuples(st.integers(1, 10), st.integers(1, 10)))
def test_same_convolution_padding(filter_shape):
    padding = helpers._same_convolution_padding(filter_shape)
    assert (6, 6, 1) == helpers.convolution_output_dimension(
        (6, 6), filter_shape, (1, 1), padding, 1
    )
