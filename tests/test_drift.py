
import numpy as np
import pytest

from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation

from microscopy_preprocessing import correct_stack_drift


# ---------------------------------------------------------
# What each test checks
# ---------------------------------------------------------

'''
test_output_shape	                    Stack dimensions remain unchanged
test_output_dtype_float32	            Float32 remains float32
test_output_dtype_integer	            Integer input becomes floating point
test_known_integer_shifts	            Integer drift is recovered correctly
test_known_subpixel_shifts	            Subpixel drift is recovered accurately
test_corrected_images_are_aligned	    Residual drift is approximately zero
test_custom_reference_image	            External reference is used correctly
Input validation tests	                Invalid inputs raise ValueError
'''


# ---------------------------------------------------------
# Test image generation
# ---------------------------------------------------------

def create_test_image(shape=(128, 128), seed=42):
    """
    Generate a reproducible synthetic microscopy image
    with spatially distributed structures.
    """

    rng = np.random.default_rng(seed)

    image = rng.normal(0, 1, shape).astype(np.float32)

    # Add Gaussian structures.
    h, w = shape
    y, x = np.mgrid[:h, :w]

    for _ in range(40):
        cx = rng.uniform(10, w - 10)
        cy = rng.uniform(10, h - 10)
        sigma = rng.uniform(1.5, 5)
        amplitude = rng.uniform(20, 100)

        image += amplitude * np.exp(
            -((x - cx)**2 + (y - cy)**2) / (2 * sigma**2)
        )

    return image


def create_shifted_stack(reference, shifts):
    """
        Generate a stack by applying known shifts to a reference image.

        Parameters
        ----------
        reference : ndarray
            Reference image.

        shifts : list of tuples
            Applied shifts in (dy, dx) format.

        Returns
        -------
        ndarray
            Shifted image stack.
        """

    return np.stack([
        shift(
            reference,
            shift=displacement,
            order=1,
            mode="wrap"
        )
        for displacement in shifts
    ])



# ---------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------

def test_output_shape():
    """Output stack must have the same dimensions as input."""

    reference = create_test_image()
    stack = create_shifted_stack(
        reference,
        [(0, 0), (2, -3), (-1, 4)]
    )

    corrected, _ = correct_stack_drift(stack)

    assert corrected.shape == stack.shape


def test_output_dtype_float32():
    """Float32 input should produce float32 output."""

    reference = create_test_image()
    stack = create_shifted_stack(
        reference,
        [(0, 0), (2, -3)]
    ).astype(np.float32)

    corrected, _ = correct_stack_drift(stack)

    assert corrected.dtype == np.float32


def test_output_dtype_integer():
    """Integer input should produce floating-point output."""

    reference = create_test_image()
    stack = create_shifted_stack(
        reference,
        [(0, 0), (2, -3)]
    )

    stack = np.round(stack).astype(np.uint16)

    corrected, _ = correct_stack_drift(stack)

    assert np.issubdtype(corrected.dtype, np.floating)


# ---------------------------------------------------------
# Drift estimation
# ---------------------------------------------------------

def test_known_integer_shifts():
    """Known integer shifts should be correctly estimated."""

    reference = create_test_image()

    applied_shifts = [
        (0, 0),
        (3, -4),
        (-5, 2),
        (4, 6),
    ]

    stack = create_shifted_stack(reference, applied_shifts)

    corrected, estimated_shifts = correct_stack_drift(
        stack
    )

    expected_shifts = -np.asarray(applied_shifts)

    np.testing.assert_allclose(
        estimated_shifts,
        expected_shifts,
        atol=0.05
    )


def test_known_subpixel_shifts():
    """Subpixel shifts should be estimated accurately."""

    reference = create_test_image()

    applied_shifts = [
        (0, 0),
        (0.25, 0.75),
        (0.50, -0.50),
        (1.35, -2.65),
        (-3.25, 1.75),
        (2.50, 3.25),
        (-4.75, -2.25),
        (5.50, 4.50),
    ]   

    stack = create_shifted_stack(reference, applied_shifts)

    corrected, estimated_shifts = correct_stack_drift(
        stack,
        upsample_factor=100
    )

    expected_shifts = -np.asarray(applied_shifts)

    np.testing.assert_allclose(
        estimated_shifts,
        expected_shifts,
        atol=0.2
    )


def test_corrected_images_are_aligned():
    """Corrected images should align with the reference image."""

    reference = create_test_image()

    applied_shifts = [
        (0, 0),
        (2.5, -3.25),
        (-4.0, 1.5),
    ]

    stack = create_shifted_stack(reference, applied_shifts)

    corrected, _ = correct_stack_drift(stack)

    for img in corrected:

        residual_shift, _, _ = phase_cross_correlation(
            reference,
            img,
            upsample_factor=100
        )

        np.testing.assert_allclose(
            residual_shift,
            (0, 0),
            atol=0.15
        )


# ---------------------------------------------------------
# Reference image handling
# ---------------------------------------------------------

def test_custom_reference_image():
    """A separately supplied reference image should be used."""

    reference = create_test_image()

    applied_shifts = [
        (2, -3),
        (-4, 5),
        (3, 2),
    ]

    stack = create_shifted_stack(reference, applied_shifts)

    corrected, estimated_shifts = correct_stack_drift(
        stack,
        reference_image=reference
    )

    expected_shifts = -np.asarray(applied_shifts)

    np.testing.assert_allclose(
        estimated_shifts,
        expected_shifts,
        atol=0.05
    )


# ---------------------------------------------------------
# Input validation
# ---------------------------------------------------------

def test_invalid_2d_input():
    """A single 2D image should raise a ValueError."""

    image = create_test_image()

    with pytest.raises(ValueError):
        correct_stack_drift(image)


def test_invalid_1d_input():
    """A 1D array should raise a ValueError."""

    image = np.zeros(128)

    with pytest.raises(ValueError):
        correct_stack_drift(image)


def test_single_image_stack():
    """A stack containing only one image should raise a ValueError."""

    stack = create_test_image()[np.newaxis, :, :]

    with pytest.raises(ValueError):
        correct_stack_drift(stack)


def test_mismatched_reference_shape():
    """Reference and stack images must have matching dimensions."""

    stack = np.zeros((3, 128, 128), dtype=np.float32)
    reference = np.zeros((64, 64), dtype=np.float32)

    with pytest.raises(ValueError):
        correct_stack_drift(
            stack,
            reference_image=reference
        )


def test_nan_input():
    """Input containing NaN values should raise a ValueError."""

    stack = np.zeros((3, 128, 128), dtype=np.float32)
    stack[1, 20, 20] = np.nan

    with pytest.raises(ValueError):
        correct_stack_drift(stack)


def test_inf_input():
    """Input containing infinite values should raise a ValueError."""

    stack = np.zeros((3, 128, 128), dtype=np.float32)
    stack[1, 20, 20] = np.inf

    with pytest.raises(ValueError):
        correct_stack_drift(stack)




def test_shift_output_shape_and_dtype():
    """Shifts must have shape (n, 2) and float64 dtype."""

    reference = create_test_image()

    stack = create_shifted_stack(
        reference,
        [(0, 0), (2, -3), (-1, 4)]
    )

    corrected, shifts = correct_stack_drift(stack)

    assert shifts.shape == (len(stack), 2)
    assert shifts.dtype == np.float64

