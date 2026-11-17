import numpy as np
from numpy.fft import fft, fftshift, ifft, ifft2, ifftshift, irfftn, rfftn
from numpy.typing import ArrayLike, NDArray
from collections.abc import Sequence


# ---------------------------------------------------------
# psf/otf edge dapming | edgetaper 
# ---------------------------------------------------------


def psf_from_otf(otf: np.ndarray, size: int, otf_power: float = 1.0) -> np.ndarray:
    """
    Build a small, normalized PSF kernel from an OTF for edge tapering.

    Parameters
    ----------
    otf : np.ndarray
        2D centered OTF (DC at index n // 2), on the same pixel grid as the
        image to be tapered. Only its magnitude is used.
    size : int
        Half-width of the kernel in pixels; output shape is (2*size+1, 2*size+1).
    otf_power : float
        Exponent applied to |OTF|. Larger values give a broader PSF and
        therefore a smoother taper.

    Returns
    -------
    np.ndarray
        PSF kernel (float64) summing to 1.
    """
    otf = np.asarray(otf)
    if otf.ndim != 2:
        raise ValueError(f"otf must be 2D, got {otf.ndim}D")
    if size < 1:
        raise ValueError("size must be >= 1")

    h, w = otf.shape
    if size > min(h, w) // 2 - 1:
        raise ValueError(f"size={size} too large for OTF of shape {otf.shape}")

    psf = fftshift(ifft2(ifftshift(np.abs(otf) ** otf_power))).real

    cy, cx = h // 2, w // 2
    psf = psf[cy - size : cy + size + 1, cx - size : cx + size + 1]

    total = psf.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("PSF derived from OTF has a non-positive sum; check the OTF")
    return psf / total  # normalize AFTER cropping





def edgetaper(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """
    Edge-taper an image by blending it with a PSF-blurred copy near the borders.

    Parameters
    ----------
    image : np.ndarray
        n-D input image.
    psf : np.ndarray
        PSF with the same ndim as `image`, strictly smaller than the image on
        every non-singleton axis, centered at shape // 2. Axes where the PSF
        has length 1 are not tapered.

    Returns
    -------
    np.ndarray
        float32 array, same shape as `image`.
    """
    image = np.asarray(image, dtype=np.float32)
    psf = np.asarray(psf, dtype=np.float32)

    if image.ndim != psf.ndim:
        raise ValueError("image and psf must have the same number of dimensions")
    for ax, (s, p) in enumerate(zip(image.shape, psf.shape)):
        if p > 1 and p >= s:
            raise ValueError(
                f"psf size ({p}) must be smaller than image size ({s}) on axis {ax}"
            )

    total = psf.sum()
    if total <= 0:
        raise ValueError("psf must have a positive sum")
    psf = psf / total

    # alpha: 0 at the borders, 1 in the interior (separable, via broadcasting)
    alpha = np.ones((1,) * image.ndim, dtype=np.float32)
    for ax, (n, p) in enumerate(zip(image.shape, psf.shape)):
        if p == 1:
            continue
        other = tuple(a for a in range(psf.ndim) if a != ax)
        proj = psf.sum(axis=other)

        z = ifft(np.abs(fft(proj, n=n - 1)) ** 2).real
        z = np.append(z, z[0])
        z /= z.max()

        shape = [1] * image.ndim
        shape[ax] = n
        alpha = alpha * (1.0 - z).reshape(shape).astype(np.float32)

    # blur: zero-pad PSF and move its center to the origin (psf2otf)
    padded = np.zeros(image.shape, dtype=np.float32)
    padded[tuple(slice(0, p) for p in psf.shape)] = psf
    padded = np.roll(
        padded,
        shift=[-(p // 2) for p in psf.shape],
        axis=tuple(range(image.ndim)),
    )
    axes = tuple(range(image.ndim))
    blurred = irfftn(rfftn(image) * rfftn(padded), s=image.shape, axes=axes)

    out = alpha * image + (1.0 - alpha) * blurred
    return np.clip(out, image.min(), image.max()).astype(np.float32)



def edgetaper_from_otf(
    image: np.ndarray,
    otf: np.ndarray,
    size: int = 30,
    otf_power: float = 3.0,
) -> np.ndarray:
    """Convenience wrapper: build a PSF from `otf` and edge-taper `image`."""
    return edgetaper(image, psf_from_otf(otf, size=size, otf_power=otf_power))




# ---------------------------------------------------------
# 1/r^power kernel 
# ---------------------------------------------------------


def _r_power_kernel(
    shape: tuple[int, ...], rborder: NDArray[np.integer], power: float
) -> NDArray[np.float64]:
    """Centered kernel max(1/r - 1/R, 0)**power, R = sqrt(2)*||rborder||."""
    grids = np.ogrid[tuple(slice(0, n) for n in shape)]
    r2 = np.zeros(shape, dtype=np.float64)
    for g, n in zip(grids, shape):
        r2 += (g - n // 2) ** 2
    r = np.sqrt(r2)
    r[tuple(n // 2 for n in shape)] = np.inf  # 1/inf = 0 -> kernel is 0 at center
    big_r = np.sqrt(2.0) * np.linalg.norm(rborder)
    return np.maximum(1.0 / r - 1.0 / big_r, 0.0) ** power


def damp_edge_outside(
    image: ArrayLike,
    border: float | Sequence[float] = 0.1,
    kernel: ArrayLike | None = None,
    usepixels: int = 2,
    power: float = 3.0,
) -> NDArray[np.floating]:
    """
    Extrapolate an image outward by normalized convolution with a 1/r^power kernel.

    Output is LARGER than the input: shape = image.shape + ceil(border * image.shape).
    The original image is placed in the center of the output; the new border
    region is filled with a smooth, weighted average of the outermost pixels.

    Parameters
    ----------
    image : n-D real array.
    border : float or sequence of float
        Fraction of the image size to add (total, split around the center).
        A sequence gives one value per axis.
    kernel : optional array of shape ``output_shape``, centered at ``shape // 2``.
        Default: max(1/r - 1/R, 0)**power with R = sqrt(2) * ||added pixels||.
    usepixels : int
        Number of outermost pixels per edge used as the source. 0 uses all pixels.
    power : float
        Exponent of the default kernel.

    Returns
    -------
    np.ndarray, shape ``image.shape + ceil(border * image.shape)``.
    float32 for integer input, otherwise the input's floating dtype.
    """
    image = np.asarray(image)
    if np.iscomplexobj(image):
        raise TypeError("complex images are not supported")
    out_dtype = image.dtype if np.issubdtype(image.dtype, np.floating) else np.float32
    img = image.astype(np.float64)

    shape = np.array(img.shape)
    rborder = np.ceil(np.broadcast_to(border, shape.shape) * shape).astype(int)
    if np.any(rborder < 0) or not np.any(rborder > 0):
        raise ValueError("border must be non-negative and add at least one pixel")
    new_shape = tuple(int(s) for s in shape + rborder)

    kernel_arr: NDArray[np.float64]
    if kernel is None:
        kernel_arr = _r_power_kernel(new_shape, rborder, power)
    else:
        kernel_arr = np.asarray(kernel, dtype=np.float64)
        if kernel_arr.shape != new_shape:
            raise ValueError(
                f"kernel must have shape {new_shape}, got {kernel_arr.shape}"
            )

    # source data and weights; keep only the outer `usepixels` ring
    data = img.copy()
    weights = np.ones_like(img)
    if usepixels > 0:
        if np.any(shape - 2 * usepixels <= 0):
            raise ValueError("usepixels too large for this image")
        inner = tuple(slice(usepixels, n - usepixels) for n in shape)
        data[inner] = 0.0
        weights[inner] = 0.0

    # embed in the larger array, aligning centers (n // 2) of source and target
    offs = tuple(int(nn // 2 - n // 2) for nn, n in zip(new_shape, shape))
    region = tuple(slice(o, o + n) for o, n in zip(offs, shape))
    data_p = np.zeros(new_shape)
    weights_p = np.zeros(new_shape)
    data_p[region] = data
    weights_p[region] = weights

    # normalized convolution (kernel center moved to the origin)
    axes = tuple(range(img.ndim))
    k_f = rfftn(ifftshift(kernel_arr), axes=axes)
    num = irfftn(rfftn(data_p, axes=axes) * k_f, s=new_shape, axes=axes)
    den = irfftn(rfftn(weights_p, axes=axes) * k_f, s=new_shape, axes=axes)

    out = np.zeros(new_shape)
    np.divide(num, den, out=out, where=den > 1e-9 * den.max())

    out[region] = img  # restore the original in the middle
    return out.astype(out_dtype)