import tifffile
from pathlib import Path
import numpy as np
from typing import Sequence
from numpy import typing as npt

# OME metadata is not supported in this function. Use tifffile.imwrite with the appropriate metadata if needed.
# TODO: Add OME metadata support in the future if required.



def read_tiff_stack(
    path: str | Path,
    filename: str,
    dtype: type[np.generic] | None = None,
) -> np.ndarray:
    """
    Read a TIFF image or stack.

    Parameters
    ----------
    path : str | Path
        Directory containing the TIFF file.
    filename : str
        TIFF filename, with or without extension.
    dtype : type[np.generic] | None, optional
        Optional output data type. If None, preserve original dtype.

    Returns
    -------
    np.ndarray
        Loaded image data.
    """

    filename = str(filename)

    if not filename.lower().endswith((".tif", ".tiff")):
        filename += ".tiff"

    file_path = Path(path) / filename

    if not file_path.is_file():
        raise FileNotFoundError(f"TIFF file not found: {file_path}")

    stack = tifffile.imread(file_path)

    if dtype is not None:
        stack = stack.astype(dtype, copy=False)

    return stack






def save_tiff_stack(
    stack: np.ndarray,
    path: str | Path,
    filename: str,
    indices: Sequence[int] | npt.NDArray[np.integer] | None = None,
    axes: str | None = None,
    dtype: type[np.generic] = np.float32,
    imagej: bool = True,
) -> Path:
    """
    Save a microscopy image stack as a TIFF file.

    Parameters
    ----------
    stack : np.ndarray
        Input image stack.
    path : str | Path
        Output directory.
    filename : str
        Output filename, with or without TIFF extension.
    indices : Sequence[int] | None, optional
        Indices along the first axis to save.
        If None, the entire stack is saved.
    axes : str | None, optional
        Image axes metadata (e.g. 'TYX', 'ZYX').
        If None, no axes metadata is written.
    dtype : np.dtype, optional
        Output data type. Defaults to np.float32.
    imagej : bool, optional
        Whether to write an ImageJ-compatible TIFF.

    Returns
    -------
    Path
        Path to the saved TIFF file.
    """

    # Validate input
    if not isinstance(stack, np.ndarray):
        raise TypeError("stack must be a numpy array.")

    if stack.ndim < 2:
        raise ValueError("stack must have at least two dimensions.")

    if stack.size == 0:
        raise ValueError("Cannot save an empty stack.")

    # Handle filename extension
    filename = str(filename)
    if not filename.lower().endswith((".tif", ".tiff")):
        filename += ".tiff"

    # Select stack indices
    if indices is not None:
        if stack.ndim < 3:
            raise ValueError(
                "Index selection requires a stack with at least 3 dimensions."
            )

        index_array = np.asarray(indices)

        if index_array.ndim != 1 or not np.issubdtype(index_array.dtype, np.integer):
            raise TypeError("indices must be a one-dimensional sequence of integers.")

        if len(index_array) == 0:
            raise ValueError("indices cannot be empty.")

        if np.any(index_array < 0) or np.any(index_array >= stack.shape[0]):
            raise IndexError(
                f"indices must be between 0 and {stack.shape[0] - 1}."
            )

        stack = stack[index_array]

    # Validate axes metadata
    if axes is not None:
        if len(axes) != stack.ndim:
            raise ValueError(
                f"axes '{axes}' does not match stack dimensionality ({stack.ndim})."
            )

    # Create output directory
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)

    output_path = output_path / filename

    # Convert data type
    stack = stack.astype(dtype, copy=False)

    # Write TIFF
    metadata = {"axes": axes} if axes is not None else None

    tifffile.imwrite(
        output_path,
        stack,
        imagej=imagej,
        metadata=metadata,
    )

    return output_path



