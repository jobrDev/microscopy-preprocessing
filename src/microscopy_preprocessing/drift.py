import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation



def _prepare_drift_stack(images):
    """
    Validate and prepare a microscopy image stack.
    """
    images = np.asarray(images)

    if images.ndim == 2:
        raise ValueError(
            "Drift correction requires an image stack or a separate reference image."
        )

    if images.ndim != 3:
        raise ValueError(
            "Input must be a 3D stack with shape (n, h, w)."
        )

    if images.shape[0] < 2:
        raise ValueError(
            "Drift correction requires at least two images."
        )

    if not np.all(np.isfinite(images)):
        raise ValueError("Input images contain NaN or infinite values.")

    return images






def _estimate_image_shift(reference, moving, upsample_factor=100):
    """
    Estimate subpixel displacement between two 2D images.

    Returns
    -------
    shift_est : ndarray
        Estimated displacement (dy, dx), i.e. the shift to apply
        to the moving image to align it with the reference.
    """
    if reference.shape != moving.shape:
        raise ValueError("Reference and moving images must have identical shapes.")

    shift_est, _, _ = phase_cross_correlation(
        reference,
        moving,
        upsample_factor=upsample_factor
    )

    return shift_est





def correct_stack_drift(
    image_stack,
    reference_image=None,
    interpolation_order=1,
    upsample_factor=100
):
    """
    Correct lateral drift in a conventional microscopy image stack
    using phase cross-correlation.

    Parameters
    ----------
    image_stack : ndarray
        Image stack with shape (n, h, w).

    reference_image : ndarray, optional
        Reference image with shape (h, w).
        If None, the first image of the stack is used.

    interpolation_order : int, optional
        Spline interpolation order used for image shifting (default: 1).

    upsample_factor : int, optional
        Upsampling factor for subpixel shift estimation (default: 100).

    Returns
    -------
    corrected : ndarray
        Drift-corrected image stack with shape (n, h, w).

    shifts : ndarray
        Estimated correction shifts in pixels, with shape (n, 2).
        Column 0: vertical displacement (dy).
        Column 1: horizontal displacement (dx).
    """

    stack = _prepare_drift_stack(image_stack)

    # Select reference image.
    if reference_image is None:
        reference = stack[0]
    else:
        reference = np.asarray(reference_image)

    if reference.ndim != 2 or reference.shape != stack.shape[1:]:
        raise ValueError("Reference image must have shape (h, w) matching the stack.")

    if not np.all(np.isfinite(reference)):
        raise ValueError("Reference image contains NaN or infinite values.")

    # Prepare output.
    output_dtype = np.result_type(stack.dtype, np.float32)
    corrected = np.empty(stack.shape, dtype=output_dtype)
    shifts = np.zeros((len(stack), 2), dtype=np.float64)

    # Process each image.
    for i, img in enumerate(stack):

        # Estimate displacement relative to reference.
        shift_est = _estimate_image_shift(
            reference,
            img,
            upsample_factor=upsample_factor
        )

        shifts[i] = shift_est

        # Apply displacement to original image.
        corrected[i] = shift(
            img.astype(output_dtype, copy=False),
            shift_est,
            order=interpolation_order
        )

    return corrected, shifts




