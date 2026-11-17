import numpy as np
import pytest
from microscopy_preprocessing import edgetaper, edgetaper_from_otf, psf_from_otf
from microscopy_preprocessing import damp_edge_outside


@pytest.fixture
def gauss_psf():
    yy, xx = np.mgrid[-16:17, -16:17]
    return np.exp(-(xx**2 + yy**2) / (2 * 3.0**2)).astype(np.float32)


@pytest.fixture
def otf_100x140():
    fy, fx = np.meshgrid(
        np.fft.fftshift(np.fft.fftfreq(100)),
        np.fft.fftshift(np.fft.fftfreq(140)),
        indexing="ij",
    )
    return np.clip(1 - np.hypot(fy, fx) / 0.25, 0, None)


def test_blur_is_not_shifted(gauss_psf):
    delta = np.zeros((128, 128), np.float32)
    delta[64, 64] = 1.0
    out = edgetaper(delta, gauss_psf)
    assert np.unravel_index(np.argmax(out), out.shape) == (64, 64)


def test_constant_image_unchanged(gauss_psf):
    const = np.full((96, 80), 5.0, np.float32)
    assert np.allclose(edgetaper(const, gauss_psf), 5.0, atol=1e-4)


def test_wraparound_discontinuity_reduced(gauss_psf):
    rng = np.random.default_rng(0)
    img = rng.random((128, 128)).astype(np.float32) + np.linspace(0, 10, 128)[None, :]
    out = edgetaper(img, gauss_psf)
    before = np.abs(img[:, 0] - img[:, -1]).mean()
    after = np.abs(out[:, 0] - out[:, -1]).mean()
    assert after < before


def test_output_shape_dtype_and_range(gauss_psf):
    rng = np.random.default_rng(1)
    img = rng.random((90, 70)).astype(np.float64)
    out = edgetaper(img, gauss_psf)
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert out.min() >= img.min() - 1e-6 and out.max() <= img.max() + 1e-6


def test_singleton_axis_is_not_tapered(gauss_psf):
    rng = np.random.default_rng(2)
    stack = rng.random((3, 64, 64)).astype(np.float32)
    out = edgetaper(stack, gauss_psf[None])
    assert out.shape == stack.shape
    # each slice should match the 2D result
    for k in range(3):
        assert np.allclose(out[k], edgetaper(stack[k], gauss_psf), atol=1e-5)


def test_psf_from_otf_shape_and_normalization(otf_100x140):
    psf = psf_from_otf(otf_100x140, size=10, otf_power=3)
    assert psf.shape == (21, 21)
    assert np.isclose(psf.sum(), 1.0)


def test_psf_from_otf_peak_at_center(otf_100x140):
    psf = psf_from_otf(otf_100x140, size=10, otf_power=3)
    assert np.unravel_index(np.argmax(psf), psf.shape) == (10, 10)


def test_edgetaper_from_otf_non_square(otf_100x140):
    rng = np.random.default_rng(3)
    img = rng.random((100, 140))
    out = edgetaper_from_otf(img, otf_100x140, size=10, otf_power=3)
    assert out.shape == (100, 140)
    assert out.dtype == np.float32


def test_wrapper_matches_manual_chain(otf_100x140):
    rng = np.random.default_rng(4)
    img = rng.random((100, 140))
    a = edgetaper_from_otf(img, otf_100x140, size=10, otf_power=3)
    b = edgetaper(img, psf_from_otf(otf_100x140, size=10, otf_power=3))
    assert np.array_equal(a, b)


def test_psf_from_otf_size_too_large(otf_100x140):
    with pytest.raises(ValueError):
        psf_from_otf(otf_100x140, size=60)


def test_psf_from_otf_requires_2d():
    with pytest.raises(ValueError):
        psf_from_otf(np.ones((8, 8, 8)), size=2)


def test_edgetaper_ndim_mismatch(gauss_psf):
    with pytest.raises(ValueError):
        edgetaper(np.zeros((3, 64, 64)), gauss_psf)


def test_edgetaper_psf_larger_than_image(gauss_psf):
    with pytest.raises(ValueError):
        edgetaper(np.zeros((20, 20)), gauss_psf)










def _center_region(out_shape, in_shape):
    offs = [o // 2 - i // 2 for o, i in zip(out_shape, in_shape)]
    return tuple(slice(o, o + n) for o, n in zip(offs, in_shape))


@pytest.fixture
def img():
    return np.random.default_rng(0).random((91, 71)).astype(np.float32)


def test_output_shape_and_dtype(img):
    out = damp_edge_outside(img, border=0.1)
    assert out.shape == (101, 79)  # ceil(0.1*91)=10, ceil(0.1*71)=8
    assert out.dtype == np.float32
    assert not np.isnan(out).any()


def test_original_preserved_in_center(img):
    out = damp_edge_outside(img)
    assert np.array_equal(out[_center_region(out.shape, img.shape)], img)


def test_constant_image_stays_constant():
    out = damp_edge_outside(np.full((60, 50), 3.0))
    assert np.allclose(out, 3.0, atol=1e-8)


def test_extrapolation_stays_within_data_range(img):
    out = damp_edge_outside(img)
    assert out.min() >= img.min() - 1e-6
    assert out.max() <= img.max() + 1e-6


def test_flip_symmetry_checks_centering(img):
    # odd output sizes (101, 79) -> exactly symmetric about the center
    out = damp_edge_outside(img)
    flipped = damp_edge_outside(img[::-1, ::-1].copy())
    assert np.allclose(flipped, out[::-1, ::-1], atol=1e-5)


def test_nd_and_per_axis_border_and_usepixels_zero():
    vol = np.random.default_rng(1).random((20, 24, 16))
    out = damp_edge_outside(vol, border=(0.1, 0.1, 0.0), usepixels=0)
    assert out.shape == (22, 27, 16)
    assert not np.isnan(out).any()




def test_custom_kernel_shape_checked(img):
    with pytest.raises(ValueError):
        damp_edge_outside(img, kernel=np.ones((5, 5)))




def test_invalid_arguments(img):
    with pytest.raises(ValueError):
        damp_edge_outside(img, border=0.0)
    with pytest.raises(ValueError):
        damp_edge_outside(img, usepixels=60)
    with pytest.raises(TypeError):
        damp_edge_outside(img.astype(np.complex64))