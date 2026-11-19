from microscopy_preprocessing import normalize_intensity
import numpy as np
import pytest


# ---------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------

def test_output_shape_and_dtype():
    """Output must preserve the input shape and use float32."""
    dat = np.random.default_rng(42).random((5, 64, 64))

    result = normalize_intensity(dat)

    assert result.shape == dat.shape
    assert result.dtype == np.float32


def test_output_is_independent_copy():
    """The original input must not be modified."""
    dat = np.random.default_rng(42).random((4, 32, 32))
    original = dat.copy()

    result = normalize_intensity(dat)

    np.testing.assert_array_equal(dat, original)
    assert not np.shares_memory(result, dat)


def test_all_slices_have_global_mean():
    """Every normalized slice must have approximately the global mean."""
    dat = np.stack([
        np.full((20, 20), 10.0),
        np.full((20, 20), 20.0),
        np.full((20, 20), 30.0),
    ])

    result = normalize_intensity(dat)

    expected_mean = np.mean(dat)

    for i in range(result.shape[0]):
        assert np.mean(result[i]) == pytest.approx(expected_mean)


def test_global_mean_is_preserved():
    """Normalization must preserve the global mean of the stack."""
    dat = np.random.default_rng(42).uniform(1, 100, (5, 64, 64))

    result = normalize_intensity(dat)

    assert np.mean(result, dtype=np.float64) == pytest.approx(
        np.mean(dat, dtype=np.float64), rel=1e-6
    )


def test_uniform_stack_remains_unchanged():
    """A stack with identical slice means should remain unchanged."""
    dat = np.full((4, 32, 32), 50.0, dtype=np.float32)

    result = normalize_intensity(dat)

    np.testing.assert_allclose(result, dat, rtol=1e-6)


# ---------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------

def test_known_normalization_factors():
    """Check normalization against manually calculated factors."""
    dat = np.stack([
        np.full((10, 10), 10.0),
        np.full((10, 10), 20.0),
        np.full((10, 10), 30.0),
    ])

    result = normalize_intensity(dat)

    # Global mean = 20
    expected = np.stack([
        np.full((10, 10), 20.0),
        np.full((10, 10), 20.0),
        np.full((10, 10), 20.0),
    ])

    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_relative_intensity_distribution_is_preserved():
    """Normalization must scale each slice without changing its spatial structure."""
    rng = np.random.default_rng(42)
    dat = rng.uniform(1, 100, (3, 32, 32))

    result = normalize_intensity(dat)

    for i in range(dat.shape[0]):
        factor = np.mean(result[i]) / np.mean(dat[i])

        np.testing.assert_allclose(
            result[i],
            dat[i] * factor,
            rtol=1e-5,
            atol=1e-5
        )


def test_float64_input():
    """Float64 input must be accepted and converted to float32."""
    dat = np.random.default_rng(42).uniform(1, 100, (3, 32, 32)).astype(
        np.float64
    )

    result = normalize_intensity(dat)

    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))


def test_integer_input():
    """Integer input must be accepted and converted to float32."""
    dat = np.random.default_rng(42).integers(1, 100, (3, 32, 32))

    result = normalize_intensity(dat)

    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))


# ---------------------------------------------------------
# Input validation
# ---------------------------------------------------------

@pytest.mark.parametrize("dat", [
    np.ones((32, 32)),          # 2D input
    np.ones((2, 3, 4, 5)),      # 4D input
    np.array(5.0),              # Scalar
    np.ones((10,)),             # 1D input
])
def test_invalid_dimensions(dat):
    """Input must be a three-dimensional stack."""
    with pytest.raises(ValueError):
        normalize_intensity(dat)


@pytest.mark.parametrize("shape", [
    (0, 10, 10),
    (3, 0, 10),
    (3, 10, 0),
])
def test_empty_dimensions(shape):
    """Empty dimensions must raise ValueError."""
    dat = np.empty(shape)

    with pytest.raises(ValueError, match="empty dimension"):
        normalize_intensity(dat)


@pytest.mark.parametrize("invalid_value", [
    np.nan,
    np.inf,
    -np.inf,
])
def test_non_finite_values(invalid_value):
    """NaN and infinite values must be rejected."""
    dat = np.ones((3, 10, 10), dtype=np.float32)
    dat[0, 0, 0] = invalid_value

    with pytest.raises(ValueError, match="NaN or infinite"):
        normalize_intensity(dat)


def test_negative_intensities():
    """Negative intensity values must be rejected."""
    dat = np.ones((3, 10, 10), dtype=np.float32)
    dat[0, 0, 0] = -1

    with pytest.raises(ValueError, match="negative"):
        normalize_intensity(dat)


def test_zero_global_mean():
    """A stack with zero global mean must be rejected."""
    dat = np.zeros((3, 10, 10), dtype=np.float32)

    with pytest.raises(ValueError, match="Global mean"):
        normalize_intensity(dat)


def test_zero_mean_slice():
    """Individual slices with zero mean must be rejected."""
    dat = np.ones((3, 10, 10), dtype=np.float32)
    dat[1] = 0

    with pytest.raises(ValueError, match="Zero-mean slices"):
        normalize_intensity(dat)


def test_error_reports_correct_zero_mean_slice():
    """Error message must identify the correct zero-mean slice."""
    dat = np.ones((4, 10, 10), dtype=np.float32)
    dat[1] = 0
    dat[3] = 0

    with pytest.raises(ValueError, match=r"\[1, 3\]"):
        normalize_intensity(dat)


@pytest.mark.parametrize("dat", [
    ["a", "b", "c"],
    np.array([["a"]], dtype=object),
])
def test_non_numeric_input(dat):
    """Non-numerical input must raise TypeError."""
    with pytest.raises(TypeError):
        normalize_intensity(dat)


def test_float32_overflow_input():
    """Values exceeding float32 range must be rejected."""
    dat = np.full((2, 10, 10), 1e100, dtype=np.float64)

    with pytest.raises(ValueError, match="NaN or infinite"):
        normalize_intensity(dat)


# ---------------------------------------------------------
# Edge cases
# ---------------------------------------------------------

def test_single_slice_stack():
    """A stack containing only one valid slice should remain unchanged."""
    dat = np.random.default_rng(42).uniform(1, 100, (1, 32, 32))

    result = normalize_intensity(dat)

    np.testing.assert_allclose(result, dat, rtol=1e-6)


def test_single_pixel_slices():
    """Normalization must also work with 1x1 image slices."""
    dat = np.array([
        [[10]],
        [[20]],
        [[30]],
    ])

    result = normalize_intensity(dat)

    expected = np.array([
        [[20]],
        [[20]],
        [[20]],
    ], dtype=np.float32)

    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_zero_pixels_are_allowed():
    """Individual zero-valued pixels are valid if the slice mean is positive."""
    dat = np.ones((3, 10, 10), dtype=np.float32)
    dat[0, 0, 0] = 0

    result = normalize_intensity(dat)

    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)