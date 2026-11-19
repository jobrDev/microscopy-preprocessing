import numpy as np
from numpy.fft import fft, fftshift, ifft, ifft2, ifftshift, irfftn, rfftn
from numpy.typing import ArrayLike, NDArray
from collections.abc import Sequence


# ---------------------------------------------------------
# psf/otf edge dapming | edgetaper 
# ---------------------------------------------------------



def psf_from_otf(otf: np.ndarray, size: int, otf_power: float = 1.0) -> np.ndarray:
    r"""
    Construct a normalized point spread function (PSF) kernel from an
    optical transfer function (OTF).

    The magnitude of the OTF is raised to the specified power and
    transformed back into the spatial domain using an inverse Fourier
    transform. The resulting PSF is cropped around its center to obtain
    a smaller kernel suitable for edge tapering.

    The OTF is expected to be centered, meaning that its zero-frequency
    (DC) component is located at index ``(height // 2, width // 2)``.

    Mathematically, the PSF is obtained as

    .. math::

        P(y,x) =
        \mathcal{F}^{-1}
        \left\{
            |\mathrm{OTF}(u,v)|^{p}
        \right\}

    where :math:`p` is the specified ``otf_power`` and
    :math:`\mathcal{F}^{-1}` denotes the inverse two-dimensional
    Fourier transform.

    The PSF is then cropped around its center and normalized such that
    its sum equals one:

    .. math::

        P_{\mathrm{normalized}}(y,x) =
        \frac{P(y,x)}
        {\sum_{y,x} P(y,x)}

    Parameters
    ----------
    otf : np.ndarray
        Centered two-dimensional optical transfer function with shape
        ``(height, width)``. Only its magnitude is used.

    size : int
        Half-width of the cropped PSF kernel in pixels. The resulting
        kernel has shape ``(2 * size + 1, 2 * size + 1)``.

    otf_power : float, optional
        Exponent applied to the magnitude of the OTF before the inverse
        Fourier transform. Larger values generally produce a broader
        spatial-domain PSF and consequently a smoother edge taper.
        Default is 1.0.

    Returns
    -------
    np.ndarray
        Normalized PSF kernel with shape ``(2 * size + 1, 2 * size + 1)``
        and dtype ``float64``. The kernel sums to one.
    
    Raises
    ------
    ValueError
        If the OTF is not two-dimensional, if the requested kernel size
        is invalid for the OTF dimensions, or if the resulting PSF has
        a non-finite or non-positive sum.
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
    r"""
    Reduce boundary discontinuities in an image by blending it with a
    PSF-blurred version of itself.

    Edge tapering reduces strong intensity transitions at image boundaries,
    which can otherwise introduce ringing artifacts during subsequent
    Fourier-based image processing, particularly deconvolution.

    A spatially varying blending coefficient is calculated from the PSF.
    The original image is retained in the interior, while the contribution
    of the blurred image increases towards the boundaries.

    Mathematically, the edge-tapered image is calculated as

    .. math::

        I_{\mathrm{tapered}}(\mathbf{r}) =
        \alpha(\mathbf{r}) I(\mathbf{r})
        +
        [1-\alpha(\mathbf{r})]
        (I * P)(\mathbf{r})

    where :math:`I` is the original image, :math:`P` is the normalized
    PSF, :math:`*` denotes convolution, and :math:`\alpha` is the
    spatially varying blending coefficient.

    The blending coefficient approaches one in the interior and zero
    towards the image boundaries:

    .. math::

        \alpha(\mathbf{r}) \approx
        \begin{cases}
            1, & \text{interior} \\
            0, & \text{boundaries}
        \end{cases}

    The convolution is performed efficiently in Fourier space using
    zero-padding and a shift of the PSF center to the origin.

    Parameters
    ----------
    image : np.ndarray
        Input image with arbitrary dimensionality. The image is converted
        to ``float32`` before processing.

    psf : np.ndarray
        Point spread function with the same number of dimensions as
        ``image``. The PSF must be strictly smaller than the image along
        every axis with a PSF dimension greater than one. Its center is
        assumed to be located at ``psf.shape // 2``.

        Axes along which the PSF has length one are not tapered.

    Returns
    -------
    np.ndarray
        Edge-tapered image with the same shape as the input and dtype
        ``float32``. Output intensities are clipped to the minimum and
        maximum intensity values of the converted input image.

    Raises
    ------
    ValueError
        If the image and PSF have different dimensionalities, if the
        PSF is not smaller than the image along a non-singleton axis,
        or if the PSF has a non-positive sum.
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
    r"""
    Construct a centered radial kernel with inverse-distance power decay.

    The kernel is used for extrapolating image intensities into an
    extended boundary region. Its weights decrease with increasing
    distance from the kernel center and become zero beyond a specified
    radial cutoff.

    The radial kernel is defined as

    .. math::

        K(r) =
        \max\left(
            \frac{1}{r} - \frac{1}{R},
            0
        \right)^{p}

    where :math:`r` is the Euclidean distance from the kernel center,
    :math:`R` is the radial cutoff, and :math:`p` is the specified
    exponent.

    The cutoff radius is calculated from the border extension as

    .. math::

        R = \sqrt{2}\,\|\mathbf{b}\|_2

    where :math:`\mathbf{b}` contains the number of additional pixels
    introduced along each image axis.

    The kernel is centered at ``shape // 2``. Its center value is set
    to zero to avoid a singularity at :math:`r=0`.

    Parameters
    ----------
    shape : tuple of int
        Shape of the output kernel, corresponding to the dimensions of
        the extended image.

    rborder : np.ndarray
        Number of additional pixels introduced along each axis.

    power : float
        Exponent controlling the radial decay of the kernel weights.

    Returns
    -------
    np.ndarray
        Centered radial kernel with the requested shape and dtype
        ``float64``. Values are non-negative and decrease with distance
        from the kernel center.
    """

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
    r"""
    Extrapolate an image beyond its original boundaries using normalized
    convolution with a radial inverse-distance power kernel.

    The output image is larger than the input image. The original image
    is preserved in the center, while the newly introduced boundary
    region is filled using a weighted average of the outermost pixels.

    This approach provides a smooth extrapolation of image intensities
    beyond the original field of view. It can be used to reduce boundary
    discontinuities before Fourier-based image processing.

    The number of additional pixels is calculated independently for
    each axis:

    .. math::

        b_i = \lceil \beta_i N_i \rceil

    where :math:`N_i` is the original image dimension along axis :math:`i`
    and :math:`\beta_i` is the corresponding border fraction.

    The resulting output dimensions are

    .. math::

        N_{i,\mathrm{out}} = N_i + b_i

    The original image is centered within the extended output array.

    Extrapolation is performed using normalized convolution. Given
    source data :math:`D`, source weights :math:`W`, and kernel :math:`K`,
    the extrapolated intensity is calculated as

    .. math::

        I_{\mathrm{out}}(\mathbf{r}) =
        \frac{
            (D * K)(\mathbf{r})
        }{
            (W * K)(\mathbf{r})
        }

    where :math:`*` denotes convolution.

    The source weights are one in the selected boundary region and zero
    elsewhere. Consequently, only the specified outermost pixels
    contribute to the extrapolation.

    If no custom kernel is provided, a radial inverse-distance power
    kernel is generated:

    .. math::

        K(r) =
        \max\left(
            \frac{1}{r} - \frac{1}{R},
            0
        \right)^p

    The original image is restored in the central region after
    extrapolation to ensure that its pixel values remain unchanged.

    Parameters
    ----------
    image : ArrayLike
        Input real-valued image of arbitrary dimensionality.
        Complex-valued images are not supported.

    border : float or sequence of float, optional
        Fraction of the original image dimensions to add as an extension.
        A scalar applies the same fraction to every axis. A sequence
        specifies an individual fraction for each axis.

        The total number of additional pixels is split approximately
        equally between both sides of each axis through center alignment.
        Default is 0.1.

    kernel : ArrayLike, optional
        Custom extrapolation kernel with shape equal to the extended
        output image shape. The kernel is assumed to be centered at
        ``shape // 2``.

        If None, a radial inverse-distance power kernel is generated
        automatically using ``power``. Default is None.

    usepixels : int, optional
        Number of outermost pixels per edge used as the source region
        for extrapolation. The selected boundary region contributes
        to the normalized convolution.

        If zero, all original image pixels are used. Default is 2.

    power : float, optional
        Exponent controlling the radial decay of the automatically
        generated kernel. Only used when ``kernel`` is None.
        Default is 3.0.

    Returns
    -------
    np.ndarray
        Extended image with shape

        ``image.shape + ceil(border * image.shape)``.

        Integer input is converted to ``float32``. Floating-point input
        retains its original floating-point data type. The original image
        is preserved in the central region.

    Raises
    ------
    TypeError
        If the input image contains complex-valued data.

    ValueError
        If the border specification is invalid, if no additional pixels
        are introduced, if the custom kernel has an incorrect shape,
        or if ``usepixels`` is too large for the input image.
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

