
import numpy as np


def normalize_intensity(dat: np.ndarray) -> np.ndarray:
    r"""
    Normalize each slice of a 3D image stack to the global mean intensity.

    Each image slice is independently scaled such that its mean intensity
    matches the global mean intensity of the entire stack.

    All image stacks follow the convention ``(n, height, width)``, where
    ``n`` denotes the number of frames, followed by the image height and
    width.

    The global mean intensity is calculated as

    .. math::

        \mu_{\mathrm{global}} =
        \frac{1}{N H W}
        \sum_{i=1}^{N}
        \sum_{y=1}^{H}
        \sum_{x=1}^{W}
        D_i(y,x)

    The mean intensity of each individual slice is

    .. math::

        \mu_i =
        \frac{1}{H W}
        \sum_{y=1}^{H}
        \sum_{x=1}^{W}
        D_i(y,x)

    A normalization factor is calculated for each slice:

    .. math::

        f_i = \frac{\mu_{\mathrm{global}}}{\mu_i}

    Each slice is then multiplied by its corresponding normalization factor:

    .. math::

        D_{\mathrm{normalized},i}(y,x) =
        f_i \cdot D_i(y,x)

    This ensures that all slices have the same mean intensity, equal to
    the global mean intensity of the original stack.

    Parameters
    ----------
    dat : np.ndarray
        Input image stack with shape ``(n, height, width)``.
        Intensities must be finite and non-negative.

    Returns
    -------
    np.ndarray
        Normalized image stack with dtype ``float32`` and the same shape
        as the input. The original input is not modified.

    Raises
    ------
    TypeError
        If input cannot be converted to a numerical array.
    ValueError
        If input has invalid dimensions, contains non-finite or negative
        values, or contains zero-mean slices.
    """


    # Convert to float32 and create an independent copy.
    try:
        dat = np.asarray(dat, dtype=np.float32).copy()
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("Input must be convertible to a numerical array.") from exc

    # Validate dimensions.
    if dat.ndim != 3:
        raise ValueError(
            f"Expected a 3D stack (n, height, width), got {dat.shape}."
        )

    if any(dim == 0 for dim in dat.shape):
        raise ValueError(f"Input contains an empty dimension: {dat.shape}.")

    # Validate intensity values.
    if not np.all(np.isfinite(dat)):
        raise ValueError("Input contains NaN or infinite values.")

    if np.any(dat < 0):
        raise ValueError("Input contains negative intensity values.")

    # Calculate global and individual slice means.
    global_mean = float(np.mean(dat, dtype=np.float64))

    slice_means = np.mean(
        dat, axis=(1, 2), dtype=np.float64
    )

    if global_mean <= 0:
        raise ValueError("Global mean intensity must be greater than zero.")

    if np.any(slice_means <= 0):
        bad_slices = np.flatnonzero(slice_means <= 0)
        raise ValueError(
            f"Zero-mean slices detected: {bad_slices.tolist()}."
        )

    # Calculate one normalization factor per slice.
    factors = np.asarray(global_mean / slice_means, dtype=np.float32)

    if not np.all(np.isfinite(factors)):
        raise ValueError("Normalization factors contain invalid values.")

    # Apply normalization using broadcasting.
    dat *= factors.reshape(-1, 1, 1)

    return dat







