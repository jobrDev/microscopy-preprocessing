
import numpy as np
import pytest

from microscopy_preprocessing import subtract_mean_background


def generate_test_stack(
    n_measurement=5,
    n_background=3,
    height=64,
    width=64,
    background_level=100,
    noise_std=2.0,
    seed=42,
):
    """
    Generate synthetic microscopy measurement and background stacks.

    Measurement images = signal + background + Gaussian noise
    Background images  = background + Gaussian noise

    Returns
    -------
    datr : np.ndarray
        Synthetic measurement stack with shape (n_measurement, height, width).

    backgrounds : np.ndarray
        Synthetic background stack with shape (n_background, height, width).

    signal : np.ndarray
        Ground-truth signal image with shape (height, width).

    background : np.ndarray
        Ground-truth background image with shape (height, width).
    """
    rng = np.random.default_rng(seed)

    # Create a spatially varying background
    y, x = np.mgrid[:height, :width]

    background = (
        background_level
        + 0.1 * x
        + 0.2 * y
    )

    # Create a known measurement signal (Gaussian spot)
    cx, cy = width // 2, height // 2
    sigma = 5

    signal = 50 * np.exp(
        -((x - cx)**2 + (y - cy)**2) / (2 * sigma**2)
    )

    # Generate measurement frames
    datr = np.array([
        signal
        + background
        + rng.normal(0, noise_std, (height, width))
        for _ in range(n_measurement)
    ])

    # Generate background frames
    backgrounds = np.array([
        background
        + rng.normal(0, noise_std, (height, width))
        for _ in range(n_background)
    ])

    return datr, backgrounds, signal, background


# ---------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------

def test_background_subtraction():
    datr, backgrounds, signal, _ = generate_test_stack()

    corrected = subtract_mean_background(datr, backgrounds)

    # Correct output shape
    assert corrected.shape == datr.shape

    # Correct output dtype
    assert corrected.dtype == np.float32

    # Compare average corrected image against ground truth
    mean_corrected = np.mean(corrected, axis=0)

    rmse = np.sqrt(np.mean((mean_corrected - signal)**2))

    assert rmse < 1.5


def test_gaussian_noise():
    datr, backgrounds, signal, _ = generate_test_stack(
        noise_std=5.0,
        seed=123,
    )

    corrected = subtract_mean_background(datr, backgrounds)

    # Average corrected measurement frames
    mean_corrected = np.mean(corrected, axis=0)

    # Calculate residual error
    rmse = np.sqrt(np.mean((mean_corrected - signal)**2))

    # Allow for residual measurement and background noise
    assert rmse < 5.0


# ---------------------------------------------------------------------
# Background characteristics
# ---------------------------------------------------------------------

def test_constant_background():
    signal = np.ones((32, 32), dtype=np.float32) * 20
    background = np.ones((32, 32), dtype=np.float32) * 100

    datr = np.stack([signal + background] * 4)
    backgrounds = np.stack([background] * 3)

    corrected = subtract_mean_background(datr, backgrounds)

    np.testing.assert_allclose(corrected, 20.0)


def test_spatial_background():
    height, width = 32, 32

    y, x = np.mgrid[:height, :width]

    background = 50 + x * 2 + y * 0.5
    signal = np.ones((height, width)) * 10

    datr = np.stack([
        signal + background,
        signal + background,
    ])

    backgrounds = np.stack([
        background,
        background,
    ])

    corrected = subtract_mean_background(datr, backgrounds)

    np.testing.assert_allclose(corrected, 10.0)


def test_single_background_frame():
    signal = np.ones((32, 32)) * 15
    background = np.ones((32, 32)) * 80

    datr = np.stack([signal + background] * 3)
    backgrounds = background[np.newaxis, :, :]

    corrected = subtract_mean_background(datr, backgrounds)

    np.testing.assert_allclose(
        corrected,
        np.broadcast_to(signal, corrected.shape),
    )


# ---------------------------------------------------------------------
# Data types and input integrity
# ---------------------------------------------------------------------

def test_uint16_input():
    signal = np.full((32, 32), 20, dtype=np.uint16)
    background = np.full((32, 32), 100, dtype=np.uint16)

    datr = np.stack([signal + background] * 3)
    backgrounds = np.stack([background] * 2)

    corrected = subtract_mean_background(datr, backgrounds)

    assert corrected.dtype == np.float32
    np.testing.assert_allclose(corrected, 20.0)


def test_input_arrays_are_not_modified():
    datr, backgrounds, _, _ = generate_test_stack()

    datr_original = datr.copy()
    backgrounds_original = backgrounds.copy()

    subtract_mean_background(datr, backgrounds)

    np.testing.assert_array_equal(datr, datr_original)
    np.testing.assert_array_equal(backgrounds, backgrounds_original)


def test_different_number_of_frames():
    datr, backgrounds, _, _ = generate_test_stack(
        n_measurement=7,
        n_background=2,
    )

    corrected = subtract_mean_background(datr, backgrounds)

    assert corrected.shape == (7, 64, 64)


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

def test_invalid_measurement_dimensions():
    datr = np.zeros((64, 64))
    backgrounds = np.zeros((3, 64, 64))

    with pytest.raises(ValueError, match="datr"):
        subtract_mean_background(datr, backgrounds)


def test_invalid_background_dimensions():
    datr = np.zeros((5, 32, 32))
    backgrounds = np.zeros((32, 32))

    with pytest.raises(ValueError, match="background"):
        subtract_mean_background(datr, backgrounds)


def test_mismatched_spatial_dimensions():
    datr = np.zeros((5, 32, 32))
    backgrounds = np.zeros((3, 64, 64))

    with pytest.raises(ValueError, match="Spatial dimensions"):
        subtract_mean_background(datr, backgrounds)


def test_empty_background_stack():
    datr = np.zeros((5, 32, 32))
    backgrounds = np.empty((0, 32, 32))

    with pytest.raises(ValueError, match="at least one frame"):
        subtract_mean_background(datr, backgrounds)


def test_non_numeric_measurement_stack():
    datr = np.full((5, 32, 32), "invalid")
    backgrounds = np.zeros((3, 32, 32))

    with pytest.raises(TypeError, match="datr"):
        subtract_mean_background(datr, backgrounds)


def test_non_numeric_background_stack():
    datr = np.zeros((5, 32, 32))
    backgrounds = np.full((3, 32, 32), "invalid")

    with pytest.raises(TypeError, match="background"):
        subtract_mean_background(datr, backgrounds)
