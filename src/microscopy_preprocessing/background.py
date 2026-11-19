
import numpy as np


def subtract_mean_background(
    datr: np.ndarray,
    background: np.ndarray,
) -> np.ndarray:
    r"""
    Subtract the mean background image from a microscopy image stack.

    The function calculates the pixel-wise mean of a separate background
    stack and subtracts this mean background image from every frame of the
    measurement stack.

    All image stacks in this preprocessing repository follow the convention
    ``(n, height, width)``, where ``n`` is the number of frames, followed by
    the image height and width.

    The measurement and background stacks must have matching spatial
    dimensions, but can contain different numbers of frames.
    
    Mathematically, the mean background image is calculated as

    .. math::

        B_{\\mathrm{mean}}(y, x) =
        \\frac{1}{N_{\\mathrm{bg}}}
        \\sum_{j=1}^{N_{\\mathrm{bg}}}
        B_j(y, x)

    The mean background is then subtracted from every measurement frame:

    .. math::

        D_{\\mathrm{corrected},i}(y, x) =
        D_i(y, x) - B_{\\mathrm{mean}}(y, x)

    Parameters
    ----------
    datr : np.ndarray
        Raw measurement image stack with shape
        ``(n_frames, height, width)``.

    background : np.ndarray
        Background image stack with shape
        ``(n_background_frames, height, width)``.
        The spatial dimensions must match those of ``datr``.

    Returns
    -------
    np.ndarray
        Background-corrected measurement stack with shape identical to
        ``datr`` and data type ``float32``.

    Raises
    ------
    ValueError
        If either input is not a three-dimensional stack, if their spatial
        dimensions do not match, or if the background stack is empty.

    TypeError
        If either input contains non-numeric data.

    Notes
    -----
    The input arrays are not modified. Both stacks are converted to
    ``float32`` before subtraction to avoid unsigned integer underflow.
    """

    if datr.ndim != 3:
        raise ValueError(
            f"Expected datr with shape (n_frames, height, width), "
            f"got {datr.shape}."
        )

    if background.ndim != 3:
        raise ValueError(
            f"Expected background with shape "
            f"(n_frames, height, width), got {background.shape}."
        )

    if not np.issubdtype(datr.dtype, np.number):
        raise TypeError(f"Unsupported datr data type: {datr.dtype}")

    if not np.issubdtype(background.dtype, np.number):
        raise TypeError(
            f"Unsupported background data type: {background.dtype}"
        )

    if datr.shape[1:] != background.shape[1:]:
        raise ValueError(
            f"Spatial dimensions must match. "
            f"Got datr: {datr.shape[1:]}, "
            f"background: {background.shape[1:]}."
        )

    if background.shape[0] == 0:
        raise ValueError("Background stack must contain at least one frame.")

    datr = datr.astype(np.float32, copy=False)
    background = background.astype(np.float32, copy=False)

    background_mean = np.mean(background, axis=0)

    corrected_stack = datr - background_mean[None, :, :]

    return corrected_stack
