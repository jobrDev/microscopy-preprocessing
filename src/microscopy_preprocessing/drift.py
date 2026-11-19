import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation




def _prepare_drift_stack(images):
    r"""
    Validate and prepare a microscopy image stack for drift correction.

    Ensures that the input is a valid 3D image stack containing at least
    two finite-valued images.

    All image stacks follow the convention ``(n, height, width)``, where
    ``n`` denotes the number of frames, followed by the image height and
    width.

    Parameters
    ----------
    images : np.ndarray
        Input microscopy image stack with shape ``(n, height, width)``.
        All pixel values must be finite.

    Returns
    -------
    np.ndarray
        Validated image stack with its original data type and shape.

    Raises
    ------
    ValueError
        If the input is a single 2D image, is not a 3D stack, contains
        fewer than two images, or contains NaN or infinite values.
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
    r"""
    Estimate the lateral displacement between two microscopy images
    using phase cross-correlation.

    The displacement is estimated in Fourier space by comparing the
    phase information of the reference and moving images. Phase
    cross-correlation provides an estimate of the translation between
    the two images with subpixel precision.

    Mathematically, the cross-power spectrum is calculated as

    .. math::

        R(u,v) =
        \frac{
            F_{\mathrm{ref}}(u,v)
            \cdot F_{\mathrm{mov}}^*(u,v)
        }{
            |F_{\mathrm{ref}}(u,v)
            \cdot F_{\mathrm{mov}}^*(u,v)|
        }

    where :math:`F_{\mathrm{ref}}` and :math:`F_{\mathrm{mov}}` are the
    Fourier transforms of the reference and moving images, respectively,
    and :math:`^*` denotes complex conjugation.

    The inverse Fourier transform of the cross-power spectrum gives a
    correlation peak. The location of this peak determines the relative
    displacement between the two images.

    The returned displacement represents the shift that must be applied
    to the moving image to align it with the reference image.

    Parameters
    ----------
    reference : np.ndarray
        Reference image with shape ``(height, width)``.

    moving : np.ndarray
        Moving image with shape ``(height, width)``.
        Must have the same dimensions as the reference image.

    upsample_factor : int, optional
        Upsampling factor used to estimate the correlation peak with
        subpixel precision. Higher values allow finer displacement
        estimates. Default is 100.

    Returns
    -------
    np.ndarray
        Estimated correction displacement in pixels with shape ``(2,)``.
        Element 0 represents the vertical displacement ``(dy)`` and
        element 1 represents the horizontal displacement ``(dx)``.

    Raises
    ------
    ValueError
        If the reference and moving images have different shapes.
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
    r"""
    Correct lateral drift in a microscopy image stack using phase
    cross-correlation.

    The lateral displacement of each frame is estimated relative to a
    reference image. The estimated displacement is then applied to the
    corresponding frame to align it with the reference.

    All image stacks follow the convention ``(n, height, width)``, where
    ``n`` denotes the number of frames, followed by the image height and
    width.

    If no reference image is provided, the first frame of the input stack
    is used as the reference. Alternatively, an independent reference
    image can be supplied.

    The displacement of each frame is estimated using phase
    cross-correlation in Fourier space.

    For each frame :math:`i`, the estimated correction displacement is

    .. math::

        \\Delta_i = (\\Delta y_i, \\Delta x_i)

    where :math:`\\Delta y_i` and :math:`\\Delta x_i` represent the
    vertical and horizontal displacements, respectively.

    The corrected image is obtained by applying the estimated translation:

    .. math::

        D_{\\mathrm{corrected},i}(y,x) =
        D_i(y - \\Delta y_i, x - \\Delta x_i)

    The translation is applied using spline interpolation. Subpixel
    displacements are therefore supported.

    Parameters
    ----------
    image_stack : np.ndarray
        Input microscopy image stack with shape ``(n, height, width)``.
        All pixel values must be finite.

    reference_image : np.ndarray, optional
        Reference image with shape ``(height, width)``.
        If None, the first frame of the input stack is used as the
        reference.

    interpolation_order : int, optional
        Spline interpolation order used when shifting the images.
        Default is 1 (linear interpolation).

    upsample_factor : int, optional
        Upsampling factor used for subpixel displacement estimation
        during phase cross-correlation. Higher values allow finer
        displacement estimates. Default is 100.

    Returns
    -------
    corrected : np.ndarray
        Drift-corrected image stack with shape ``(n, height, width)``.
        The output data type is determined by the input data type and
        ``float32``, ensuring that floating-point interpolation is
        supported.

    shifts : np.ndarray
        Estimated correction displacements in pixels with shape ``(n, 2)``.
        Column 0 contains the vertical displacement ``(dy)`` and
        column 1 contains the horizontal displacement ``(dx)``.

        The shifts represent the translations applied to the original
        frames to align them with the reference image.

    Raises
    ------
    ValueError
        If the input stack has invalid dimensions, contains fewer than
        two images, contains non-finite values, or if the reference image
        has invalid dimensions or does not match the spatial dimensions
        of the stack.
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




