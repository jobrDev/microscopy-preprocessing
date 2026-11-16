
import numpy as np



def subtract_mean_background(datr: np.ndarray, bg_count: int, ignore_last=False) -> np.ndarray:
    """
    Subtract the mean background from the measurement images.

    Parameters
    ----------
    datr : np.ndarray
        Input grayscale stack with shape (frames, height, width).
    bg_count : int
        Number of background frames at the end of the stack.
    ignore_last : bool, default=False
        If True, discard the final frame before splitting the stack.

    Returns
    -------
    np.ndarray
        Background-corrected measurement stack as float32.
    """

    if datr.ndim != 3:
        raise ValueError(
            f"Expected a grayscale stack with shape "
            f"(frames, height, width), got {datr.shape}."
        )

    if not np.issubdtype(datr.dtype, np.number):
        raise TypeError(f"Unsupported image data type: {datr.dtype}")

    if bg_count < 1:
        raise ValueError("bg_count must be at least 1.")

    datr = datr.astype(np.float32, copy=False)

    if ignore_last:
        if len(datr) < 2:
            raise ValueError("Cannot ignore the last frame: stack is too short.")
        datr = datr[:-1]

    if bg_count >= len(datr):
        raise ValueError(f"bg_count={bg_count} leaves no measurement frames.")

    measurement_images = datr[:-bg_count]
    bg_images = datr[-bg_count:]

    bg_mean = np.mean(bg_images, axis=0)

    corrected_stack = measurement_images - bg_mean[None, :, :]

    return corrected_stack


