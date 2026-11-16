import numpy as np
from microscopy_preprocessing import subtract_mean_background
import numpy as np
import pytest





def test_background_subtraction():
    stack, signal, background = generate_test_stack()

    corrected = subtract_mean_background(stack, bg_count=3)

    # Correct output shape
    assert corrected.shape == (5, 64, 64)

    # Correct output dtype
    assert corrected.dtype == np.float32

    # Compare average corrected image against ground truth
    mean_corrected = np.mean(corrected, axis=0)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((mean_corrected - signal)**2))

    assert rmse < 1.5





def test_gaussian_noise():
    stack, signal, background = generate_test_stack(
        noise_std=5.0,
        seed=123,
    )

    corrected = subtract_mean_background(stack, bg_count=3)

    # Average corrected measurement frames
    mean_corrected = np.mean(corrected, axis=0)

    # Calculate residual error
    error = mean_corrected - signal
    rmse = np.sqrt(np.mean(error**2))

    # Allow some residual noise
    assert rmse < 5.0




    
def test_constant_background():
    signal = np.ones((32, 32), dtype=np.float32) * 20
    background = np.ones((32, 32), dtype=np.float32) * 100

    measurements = np.stack([signal + background] * 4)
    backgrounds = np.stack([background] * 3)

    stack = np.concatenate([measurements, backgrounds])

    corrected = subtract_mean_background(stack, bg_count=3)

    np.testing.assert_allclose(corrected, 20.0)






def test_spatial_background():
    height, width = 32, 32

    y, x = np.mgrid[:height, :width]

    background = 50 + x * 2 + y * 0.5
    signal = np.ones((height, width)) * 10

    measurements = np.stack([
        signal + background,
        signal + background,
    ])

    backgrounds = np.stack([
        background,
        background,
    ])

    stack = np.concatenate([measurements, backgrounds])

    corrected = subtract_mean_background(stack, bg_count=2)

    np.testing.assert_allclose(corrected, 10.0)




def test_ignore_last():
    stack, signal, background = generate_test_stack(
        n_measurement=5,
        n_background=3,
    )

    # Append an additional frame that should be ignored
    extra_frame = np.ones((1, 64, 64)) * 999

    stack = np.concatenate([stack, extra_frame])

    corrected = subtract_mean_background(
        stack,
        bg_count=3,
        ignore_last=True,
    )

    assert corrected.shape == (5, 64, 64)




def test_invalid_dimensions():
    stack = np.zeros((64, 64))

    with pytest.raises(ValueError):
        subtract_mean_background(stack, bg_count=2)


def test_invalid_bg_count():
    stack = np.zeros((5, 32, 32))

    with pytest.raises(ValueError):
        subtract_mean_background(stack, bg_count=0)


def test_no_measurement_frames():
    stack = np.zeros((5, 32, 32))

    with pytest.raises(ValueError):
        subtract_mean_background(stack, bg_count=5)


def test_ignore_last_too_short():
    stack = np.zeros((1, 32, 32))

    with pytest.raises(ValueError):
        subtract_mean_background(stack, bg_count=1, ignore_last=True)



















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
    Generate synthetic measurement and background images.

    Measurement images = signal + background + Gaussian noise
    Background images  = background + Gaussian noise
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
    measurements = np.array([
        signal + background
        + rng.normal(0, noise_std, (height, width))
        for _ in range(n_measurement)
    ])

    # Generate background frames
    backgrounds = np.array([
        background
        + rng.normal(0, noise_std, (height, width))
        for _ in range(n_background)
    ])

    stack = np.concatenate([measurements, backgrounds], axis=0)

    return stack, signal, background
