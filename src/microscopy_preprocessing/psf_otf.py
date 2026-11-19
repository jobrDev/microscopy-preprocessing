import numpy as np
from scipy.special import j1
from numpy.fft import fft2, fftshift, ifftshift, ifft2



def generate_amplitude_psf(size: int, scale: float) -> np.ndarray:
    r"""
    Generate the amplitude point spread function (PSF) of an ideal
    circular optical aperture.

    The amplitude PSF is described by the jinc function:

    .. math::

        A(r) = \frac{2 J_1(v)}{v},
        \qquad v = \mathrm{scale} \cdot r

    where :math:`J_1` is the first-order Bessel function of the
    first kind and :math:`r` is the radial distance from the
    optical axis.

    At the origin, the analytical limit is:

    .. math::

        \lim_{v \to 0} \frac{2 J_1(v)}{v} = 1

    Parameters
    ----------
    size : int
        Number of pixels along each dimension of the square PSF.

    scale : float
        Radial scaling factor controlling the width of the PSF.
        Larger values result in a narrower PSF.

    Returns
    -------
    numpy.ndarray
        Two-dimensional real-valued amplitude PSF, normalized
        such that its central maximum is 1.

    Notes
    -----
    The scale parameter is dimensionless and depends on the
    spatial sampling and optical parameters.

    The returned PSF is centered in the array.
    """

    if not isinstance(size, (int, np.integer)) or isinstance(size, (bool, np.bool_)) or size <= 0:
        raise ValueError("size must be a positive integer.")

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite number.")

    # Generate centered coordinate grid.
    coords = np.arange(size) - size // 2
    X, Y = np.meshgrid(coords, coords)

    # Calculate radial distance from the optical axis.
    radius = np.sqrt(X**2 + Y**2)

    # Calculate argument of the jinc function.
    argument = scale * radius

    # Initialize amplitude PSF.
    # The analytical value at the origin is 1.
    psf = np.ones_like(argument, dtype=float)

    # Evaluate the jinc function everywhere except at the origin.
    nonzero = argument != 0
    psf[nonzero] = (
        2 * j1(argument[nonzero]) / argument[nonzero]
    )

    # Normalize central maximum to 1.
    psf /= np.max(np.abs(psf))

    return psf





def generate_intensity_psf(size: int, scale: float) -> np.ndarray:
    r"""
    Generate the intensity point spread function (PSF) of an
    ideal circular optical aperture.

    For incoherent imaging, the intensity PSF is given by the
    squared magnitude of the amplitude PSF:

    .. math::

        I(r) = |A(r)|^2
             = \left[ \frac{2 J_1(v)}{v} \right]^2

    where :math:`J_1` is the first-order Bessel function of the
    first kind and :math:`v = \mathrm{scale} \cdot r`.

    At the origin, the intensity reaches its maximum:

    .. math::

        I(0) = 1

    Parameters
    ----------
    size : int
        Number of pixels along each dimension of the square PSF.

    scale : float
        Radial scaling factor controlling the width of the PSF.
        Larger values result in a narrower PSF.

    Returns
    -------
    numpy.ndarray
        Two-dimensional intensity PSF normalized to unit total
        energy:

        .. math::

            \sum_{x,y} I(x,y) = 1

    Notes
    -----
    The intensity PSF describes the spatial response of an
    incoherent optical imaging system.

    The amplitude PSF is calculated internally using
    :func:`generate_amplitude_psf`.
    """

    # Generate corresponding amplitude PSF.
    amplitude_psf = generate_amplitude_psf(size, scale)

    # Convert amplitude PSF to intensity PSF.
    intensity_psf = np.abs(amplitude_psf)**2

    # Normalize total intensity to 1.
    intensity_psf /= np.sum(intensity_psf)

    return intensity_psf





def generate_psf_otf(size: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Generate an intensity PSF and its corresponding incoherent
    optical transfer function (OTF).

    The intensity PSF is generated using an ideal circular
    aperture model. The incoherent OTF is calculated as the
    Fourier transform of the intensity PSF:

    .. math::

        OTF(f_x, f_y) =
        \mathcal{F}\{PSF(x,y)\}

    The OTF is normalized such that its zero-frequency
    component is equal to 1:

    .. math::

        OTF(0,0) = 1

    Parameters
    ----------
    size : int
        Number of pixels along each dimension of the square PSF.

    scale : float
        Radial scaling factor controlling the width of the PSF.

    Returns
    -------
    psf : numpy.ndarray
        Two-dimensional intensity PSF normalized to unit total
        energy.

    otf : numpy.ndarray
        Two-dimensional complex-valued incoherent OTF,
        normalized to 1 at zero spatial frequency.

    Notes
    -----
    The OTF is obtained from the Fourier transform of the
    intensity PSF, as required for incoherent imaging.

    The returned PSF and OTF are both centered in their arrays.

    The OTF retains its complex values to preserve potential
    phase information in future extensions.

    The intensity PSF is calculated internally using
    :func:`generate_intensity_psf`.
    """

    # Generate intensity PSF.
    psf = generate_intensity_psf(size, scale)

    # Move PSF origin from array center to array corner.
    psf_unshifted = ifftshift(psf)

    # Calculate incoherent OTF.
    otf = fftshift(fft2(psf_unshifted))

    # Normalize OTF at zero spatial frequency.
    center = size // 2
    otf /= otf[center, center]

    return psf, otf







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








def find_otf_cutoff(otf: np.ndarray, truncate: float = 0.01) -> int:
    r"""
    Determine the cutoff index of a two-dimensional optical transfer
    function (OTF).

    The OTF cutoff is defined as the first position along the
    central horizontal line where the magnitude of the OTF falls
    below a specified fraction of its maximum value.

    The cutoff condition is:

    .. math::

        |OTF(f_x, 0)| < \mathrm{truncate} \cdot |OTF(0,0)|

    where :math:`|OTF(0,0)|` represents the maximum OTF magnitude
    at zero spatial frequency.

    Parameters
    ----------
    otf : numpy.ndarray
        Two-dimensional optical transfer function. The zero-frequency
        component must be located at the center of the array.

    truncate : float, optional
        Relative threshold used to determine the OTF cutoff.
        For example, a value of 0.01 corresponds to 1% of the
        maximum OTF magnitude. Default is 0.01.

    Returns
    -------
    int
        Index of the first position along the positive horizontal
        frequency axis where the OTF magnitude falls below the
        specified threshold.

        The index is measured from the left edge of the array,
        matching the original array coordinates.

    Raises
    ------
    ValueError
        If the OTF is not a valid two-dimensional array, contains
        non-finite values, has zero magnitude, or if truncate
        is outside the interval (0, 1).

        Also raised if the OTF does not fall below the specified
        threshold along the central horizontal line.

    Notes
    -----
    The OTF is assumed to be centered in the array, with zero
    spatial frequency at the center.

    The cutoff is determined along the horizontal frequency axis
    only. For radially symmetric OTFs, this corresponds to the
    radial cutoff.

    The returned index is a pixel index, not a physical spatial
    frequency. Conversion to physical frequency requires knowledge
    of the image sampling and pixel size.
    """

    # ---------------------------------------------------------
    # 1. Validate input
    # ---------------------------------------------------------

    otf = np.asarray(otf)

    if otf.ndim != 2 or otf.shape[0] != otf.shape[1]:
        raise ValueError("OTF must be a two-dimensional square array.")

    if not np.all(np.isfinite(otf)):
        raise ValueError("OTF must contain only finite values.")

    # Validate cutoff threshold.
    try:
        truncate = float(truncate)
    except (TypeError, ValueError):
        raise ValueError("truncate must be a real number.") from None

    if not np.isfinite(truncate) or not 0 < truncate < 1:
        raise ValueError(
            "truncate must be a finite value between 0 and 1."
        )

    # ---------------------------------------------------------
    # 2. Extract central horizontal OTF profile
    # ---------------------------------------------------------

    # Determine the center index of the OTF.
    center = otf.shape[0] // 2

    # Extract the horizontal line passing through zero frequency.
    otf_profile = np.abs(otf[center, :])

    # Determine the maximum OTF magnitude.
    otf_max = np.max(otf_profile)

    if otf_max == 0:
        raise ValueError("OTF must have a non-zero magnitude.")

    # ---------------------------------------------------------
    # 3. Determine OTF cutoff
    # ---------------------------------------------------------

    # Define relative cutoff threshold.
    threshold = truncate * otf_max

    # Shift the profile so that zero frequency is at the beginning.
    otf_profile = np.fft.ifftshift(otf_profile)

    # Find first position where OTF magnitude falls below threshold.
    cutoff_indices = np.where(otf_profile < threshold)[0]

    if cutoff_indices.size == 0:
        raise ValueError(
            "OTF does not fall below the specified threshold."
        )

    return int(cutoff_indices[0])




