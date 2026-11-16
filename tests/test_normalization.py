from microscopy_preprocessing import normalize_intensity
import numpy as np
import pytest


def test_normalized_slice_means():
    """All slices should have approximately the same mean after normalization."""

    stack = np.random.default_rng(42).uniform(
        10, 100, size=(5, 64, 64)
    ).astype(np.float32)

    normalized = normalize_intensity(stack)

    means = np.mean(normalized, axis=(1, 2))

    assert np.allclose(means, means[0], rtol=1e-5)


def test_global_mean_preserved():
    """Normalization should preserve the global mean intensity."""

    stack = np.random.default_rng(42).uniform(
        10, 100, size=(5, 64, 64)
    ).astype(np.float32)

    normalized = normalize_intensity(stack)

    original_mean = np.mean(stack, dtype=np.float64)
    normalized_mean = np.mean(normalized, dtype=np.float64)

    assert np.isclose(original_mean, normalized_mean, rtol=1e-5)


def test_original_input_unchanged():
    """The original input array must not be modified."""

    stack = np.random.default_rng(42).uniform(
        10, 100, size=(5, 64, 64)
    ).astype(np.float32)

    original = stack.copy()

    normalize_intensity(stack)

    assert np.array_equal(stack, original)


def test_output_shape_and_dtype():
    """Output must retain the input shape and use float32."""

    stack = np.ones((5, 64, 64), dtype=np.float64)

    normalized = normalize_intensity(stack)

    assert normalized.shape == stack.shape
    assert normalized.dtype == np.float32


def test_invalid_dimensions():
    """Input must have exactly three dimensions."""

    with pytest.raises(ValueError):
        normalize_intensity(np.ones((64, 64)))

    with pytest.raises(ValueError):
        normalize_intensity(np.ones((5, 64, 64, 1)))


def test_empty_stack():
    """Empty dimensions must be rejected."""

    with pytest.raises(ValueError):
        normalize_intensity(np.empty((0, 64, 64)))

    with pytest.raises(ValueError):
        normalize_intensity(np.empty((5, 0, 64)))


def test_non_finite_values():
    """NaN and infinite values must be rejected."""

    stack = np.ones((3, 64, 64), dtype=np.float32)
    stack[0, 0, 0] = np.nan

    with pytest.raises(ValueError):
        normalize_intensity(stack)

    stack[0, 0, 0] = np.inf

    with pytest.raises(ValueError):
        normalize_intensity(stack)


def test_negative_values():
    """Negative intensities must be rejected."""

    stack = np.ones((3, 64, 64), dtype=np.float32)
    stack[0, 0, 0] = -1

    with pytest.raises(ValueError):
        normalize_intensity(stack)


def test_zero_mean_slice():
    """Slices with zero mean must be rejected."""

    stack = np.ones((3, 64, 64), dtype=np.float32)
    stack[1] = 0

    with pytest.raises(ValueError, match="Zero-mean slices"):
        normalize_intensity(stack)


def test_zero_global_mean():
    """A stack with zero global mean must be rejected."""

    stack = np.zeros((3, 64, 64), dtype=np.float32)

    with pytest.raises(ValueError, match="Global mean"):
        normalize_intensity(stack)



def test_known_normalization_factors():
    """Verify normalization against analytically known results."""

    stack = np.zeros((3, 10, 10), dtype=np.float32)

    stack[0] = 10
    stack[1] = 20
    stack[2] = 30

    normalized = normalize_intensity(stack)

    # Global mean = 20
    # Expected slice means after normalization = [20, 20, 20]

    expected = np.full((3, 10, 10), 20, dtype=np.float32)

    assert np.allclose(normalized, expected)