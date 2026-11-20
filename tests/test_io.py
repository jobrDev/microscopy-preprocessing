"""Tests for read_tiff_stack and save_tiff_stack.

Adjust the import below to match the module where the functions live.
"""

from pathlib import Path

import numpy as np
import pytest
import tifffile

from microscopy_preprocessing.io import read_tiff_stack, save_tiff_stack

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def stack_3d() -> np.ndarray:
    """A small, deterministic 3D stack (Z, Y, X) with distinct values per frame."""
    return np.arange(5 * 8 * 6, dtype=np.float32).reshape(5, 8, 6)


@pytest.fixture
def image_2d() -> np.ndarray:
    return np.arange(8 * 6, dtype=np.float32).reshape(8, 6)


def _write_raw_tiff(path: Path, data: np.ndarray) -> Path:
    """Write a TIFF directly with tifffile, independent of the code under test."""
    tifffile.imwrite(path, data)
    return path


# --------------------------------------------------------------------------- #
# read_tiff_stack
# --------------------------------------------------------------------------- #


class TestReadTiffStack:
    def test_reads_3d_stack(self, tmp_path, stack_3d):
        _write_raw_tiff(tmp_path / "stack.tiff", stack_3d)

        result = read_tiff_stack(tmp_path, "stack.tiff")

        np.testing.assert_array_equal(result, stack_3d)
        assert result.shape == stack_3d.shape

    def test_reads_2d_image(self, tmp_path, image_2d):
        _write_raw_tiff(tmp_path / "img.tiff", image_2d)

        result = read_tiff_stack(tmp_path, "img.tiff")

        np.testing.assert_array_equal(result, image_2d)
        assert result.ndim == 2

    def test_appends_tiff_extension_when_missing(self, tmp_path, stack_3d):
        _write_raw_tiff(tmp_path / "stack.tiff", stack_3d)

        result = read_tiff_stack(tmp_path, "stack")

        np.testing.assert_array_equal(result, stack_3d)

    @pytest.mark.parametrize("ext", [".tif", ".tiff", ".TIF", ".TIFF", ".Tif"])
    def test_accepts_existing_extension_case_insensitively(
        self, tmp_path, image_2d, ext
    ):
        name = f"img{ext}"
        _write_raw_tiff(tmp_path / name, image_2d)

        result = read_tiff_stack(tmp_path, name)

        np.testing.assert_array_equal(result, image_2d)

    def test_tif_file_not_found_when_extension_omitted(self, tmp_path, image_2d):
        # Without an extension the function appends ".tiff", so a file saved as
        # ".tif" is NOT found. This documents the current behaviour.
        _write_raw_tiff(tmp_path / "img.tif", image_2d)

        with pytest.raises(FileNotFoundError):
            read_tiff_stack(tmp_path, "img")

    def test_accepts_str_path(self, tmp_path, image_2d):
        _write_raw_tiff(tmp_path / "img.tiff", image_2d)

        result = read_tiff_stack(str(tmp_path), "img")

        np.testing.assert_array_equal(result, image_2d)

    def test_accepts_path_filename(self, tmp_path, image_2d):
        # filename is coerced with str(), so a Path-like name should work too.
        _write_raw_tiff(tmp_path / "img.tiff", image_2d)

        result = read_tiff_stack(tmp_path, Path("img.tiff"))  # type: ignore[arg-type]

        np.testing.assert_array_equal(result, image_2d)

    def test_preserves_original_dtype_by_default(self, tmp_path):
        data = np.arange(12, dtype=np.uint16).reshape(3, 4)
        _write_raw_tiff(tmp_path / "u16.tiff", data)

        result = read_tiff_stack(tmp_path, "u16")

        assert result.dtype == np.uint16
        np.testing.assert_array_equal(result, data)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.uint8])
    def test_dtype_conversion(self, tmp_path, dtype):
        data = np.arange(12, dtype=np.uint16).reshape(3, 4)
        _write_raw_tiff(tmp_path / "u16.tiff", data)

        result = read_tiff_stack(tmp_path, "u16", dtype=dtype)

        assert result.dtype == dtype
        np.testing.assert_array_equal(result, data.astype(dtype))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="TIFF file not found"):
            read_tiff_stack(tmp_path, "does_not_exist")

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_tiff_stack(tmp_path / "no_such_dir", "img")

    def test_directory_with_tiff_name_raises(self, tmp_path):
        # A directory that happens to be named like a TIFF is not a file.
        (tmp_path / "folder.tiff").mkdir()

        with pytest.raises(FileNotFoundError):
            read_tiff_stack(tmp_path, "folder.tiff")


# --------------------------------------------------------------------------- #
# save_tiff_stack: basic behaviour
# --------------------------------------------------------------------------- #


class TestSaveTiffStackBasics:
    def test_returns_path_and_creates_file(self, tmp_path, stack_3d):
        out = save_tiff_stack(stack_3d, tmp_path, "out")

        assert isinstance(out, Path)
        assert out == tmp_path / "out.tiff"
        assert out.is_file()

    def test_round_trip_3d(self, tmp_path, stack_3d):
        out = save_tiff_stack(stack_3d, tmp_path, "out")

        result = tifffile.imread(out)

        assert result.shape == stack_3d.shape
        np.testing.assert_array_equal(result, stack_3d)

    def test_round_trip_2d(self, tmp_path, image_2d):
        out = save_tiff_stack(image_2d, tmp_path, "out")

        result = tifffile.imread(out)

        assert result.shape == image_2d.shape
        np.testing.assert_array_equal(result, image_2d)

    def test_round_trip_with_read_tiff_stack(self, tmp_path, stack_3d):
        save_tiff_stack(stack_3d, tmp_path, "roundtrip")

        result = read_tiff_stack(tmp_path, "roundtrip")

        np.testing.assert_array_equal(result, stack_3d)

    def test_round_trip_4d(self, tmp_path):
        data = np.random.default_rng(0).random((2, 3, 4, 5)).astype(np.float32)

        out = save_tiff_stack(data, tmp_path, "out", axes="TZYX")

        np.testing.assert_array_equal(tifffile.imread(out), data)

    def test_does_not_modify_input(self, tmp_path):
        data = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        original = data.copy()

        save_tiff_stack(data, tmp_path, "out", indices=[1])

        np.testing.assert_array_equal(data, original)
        assert data.dtype == np.uint8

    def test_accepts_str_path(self, tmp_path, stack_3d):
        out = save_tiff_stack(stack_3d, str(tmp_path), "out")

        assert out.is_file()

    def test_creates_nested_output_directory(self, tmp_path, stack_3d):
        target = tmp_path / "a" / "b" / "c"
        assert not target.exists()

        out = save_tiff_stack(stack_3d, target, "out")

        assert target.is_dir()
        assert out.parent == target
        assert out.is_file()

    def test_existing_output_directory_is_fine(self, tmp_path, stack_3d):
        save_tiff_stack(stack_3d, tmp_path, "first")
        out = save_tiff_stack(stack_3d, tmp_path, "second")

        assert out.is_file()

    def test_overwrites_existing_file(self, tmp_path, stack_3d):
        save_tiff_stack(stack_3d, tmp_path, "out")
        save_tiff_stack(stack_3d * 2, tmp_path, "out")

        result = tifffile.imread(tmp_path / "out.tiff")

        np.testing.assert_array_equal(result, stack_3d * 2)


# --------------------------------------------------------------------------- #
# save_tiff_stack: filename handling
# --------------------------------------------------------------------------- #


class TestSaveTiffStackFilename:
    def test_appends_tiff_extension(self, tmp_path, stack_3d):
        out = save_tiff_stack(stack_3d, tmp_path, "name")

        assert out.name == "name.tiff"

    @pytest.mark.parametrize("name", ["a.tif", "a.tiff", "a.TIF", "a.TIFF"])
    def test_keeps_existing_tiff_extension(self, tmp_path, stack_3d, name):
        out = save_tiff_stack(stack_3d, tmp_path, name)

        assert out.name == name
        assert out.is_file()

    def test_other_extension_gets_tiff_appended(self, tmp_path, stack_3d):
        out = save_tiff_stack(stack_3d, tmp_path, "data.v2")

        assert out.name == "data.v2.tiff"

    def test_accepts_path_filename(self, tmp_path, stack_3d):
        out = save_tiff_stack(stack_3d, tmp_path, Path("out"))  # type: ignore[arg-type]

        assert out.name == "out.tiff"


# --------------------------------------------------------------------------- #
# save_tiff_stack: dtype
# --------------------------------------------------------------------------- #


class TestSaveTiffStackDtype:
    def test_default_dtype_is_float32(self, tmp_path):
        data = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)

        out = save_tiff_stack(data, tmp_path, "out")

        assert tifffile.imread(out).dtype == np.float32

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
    def test_imagej_supported_dtypes(self, tmp_path, dtype):
        data = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

        out = save_tiff_stack(data, tmp_path, "out", dtype=dtype)

        result = tifffile.imread(out)
        assert result.dtype == dtype
        np.testing.assert_array_equal(result, data.astype(dtype))

    @pytest.mark.parametrize("dtype", [np.float64, np.int32])
    def test_non_imagej_dtype_with_imagej_true_raises(self, tmp_path, dtype):
        # ImageJ-compatible TIFFs only support uint8, uint16 and float32 here.
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        with pytest.raises(ValueError):
            save_tiff_stack(data, tmp_path, "out", dtype=dtype, imagej=True)

    @pytest.mark.parametrize("dtype", [np.float64, np.int32, np.int16])
    def test_other_dtypes_work_with_imagej_false(self, tmp_path, dtype):
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        out = save_tiff_stack(data, tmp_path, "out", dtype=dtype, imagej=False)

        result = tifffile.imread(out)
        assert result.dtype == dtype
        np.testing.assert_array_equal(result, data.astype(dtype))


# --------------------------------------------------------------------------- #
# save_tiff_stack: imagej flag and axes metadata
# --------------------------------------------------------------------------- #


class TestSaveTiffStackMetadata:
    def test_imagej_true_by_default(self, tmp_path, stack_3d):
        out = save_tiff_stack(stack_3d, tmp_path, "out")

        with tifffile.TiffFile(out) as tif:
            assert tif.is_imagej

    def test_imagej_false(self, tmp_path, stack_3d):
        out = save_tiff_stack(stack_3d, tmp_path, "out", imagej=False)

        with tifffile.TiffFile(out) as tif:
            assert not tif.is_imagej

    @pytest.mark.parametrize("axes", ["TYX", "ZYX"])
    def test_axes_metadata_written_imagej(self, tmp_path, stack_3d, axes):
        out = save_tiff_stack(stack_3d, tmp_path, "out", axes=axes)

        with tifffile.TiffFile(out) as tif:
            assert tif.series[0].axes == axes
            assert tif.series[0].shape == stack_3d.shape

    def test_axes_metadata_written_without_imagej(self, tmp_path, stack_3d):
        out = save_tiff_stack(stack_3d, tmp_path, "out", axes="TYX", imagej=False)

        with tifffile.TiffFile(out) as tif:
            assert tif.series[0].axes == "TYX"

    def test_axes_for_2d_image(self, tmp_path, image_2d):
        out = save_tiff_stack(image_2d, tmp_path, "out", axes="YX")

        with tifffile.TiffFile(out) as tif:
            assert tif.series[0].axes == "YX"

    def test_no_axes_does_not_set_explicit_axes(self, tmp_path, stack_3d):
        out = save_tiff_stack(stack_3d, tmp_path, "out")

        with tifffile.TiffFile(out) as tif:
            # With no metadata, tifffile falls back to its own default axes
            # (not the user-requested T/Z).
            assert tif.series[0].axes not in ("TYX", "ZYX")

    @pytest.mark.parametrize("axes", ["YX", "TZYX", "T", ""])
    def test_axes_length_mismatch_raises(self, tmp_path, stack_3d, axes):
        with pytest.raises(ValueError, match="does not match stack dimensionality"):
            save_tiff_stack(stack_3d, tmp_path, "out", axes=axes)

    def test_axes_validated_against_selected_stack(self, tmp_path, stack_3d):
        # Selecting indices keeps ndim, so a correct-length axes string works.
        out = save_tiff_stack(stack_3d, tmp_path, "out", indices=[0, 1], axes="ZYX")

        with tifffile.TiffFile(out) as tif:
            assert tif.series[0].axes == "ZYX"
            assert tif.series[0].shape == (2, 8, 6)


# --------------------------------------------------------------------------- #
# save_tiff_stack: index selection
# --------------------------------------------------------------------------- #


class TestSaveTiffStackIndices:
    def test_selects_subset_of_frames(self, tmp_path, stack_3d):
        out = save_tiff_stack(stack_3d, tmp_path, "out", indices=[0, 2, 4])

        result = tifffile.imread(out)

        assert result.shape == (3, 8, 6)
        np.testing.assert_array_equal(result, stack_3d[[0, 2, 4]])

    def test_preserves_requested_order(self, tmp_path, stack_3d):
        out = save_tiff_stack(stack_3d, tmp_path, "out", indices=[3, 1, 2])

        np.testing.assert_array_equal(tifffile.imread(out), stack_3d[[3, 1, 2]])

    def test_allows_duplicate_indices(self, tmp_path, stack_3d):
        out = save_tiff_stack(stack_3d, tmp_path, "out", indices=[1, 1, 2])

        result = tifffile.imread(out)

        assert result.shape[0] == 3
        np.testing.assert_array_equal(result[0], result[1])
        np.testing.assert_array_equal(result[0], stack_3d[1])

    def test_accepts_numpy_array_indices(self, tmp_path, stack_3d):
        out = save_tiff_stack(stack_3d, tmp_path, "out", indices=np.array([0, 4]))

        np.testing.assert_array_equal(tifffile.imread(out), stack_3d[[0, 4]])

    def test_accepts_tuple_and_range_indices(self, tmp_path, stack_3d):
        out_tuple = save_tiff_stack(stack_3d, tmp_path, "t", indices=(0, 1))
        out_range = save_tiff_stack(stack_3d, tmp_path, "r", indices=range(2))

        np.testing.assert_array_equal(tifffile.imread(out_tuple), stack_3d[:2])
        np.testing.assert_array_equal(tifffile.imread(out_range), stack_3d[:2])

    def test_all_indices_equals_full_stack(self, tmp_path, stack_3d):
        out = save_tiff_stack(
            stack_3d, tmp_path, "out", indices=list(range(stack_3d.shape[0]))
        )

        np.testing.assert_array_equal(tifffile.imread(out), stack_3d)

    def test_single_index_keeps_frame_content(self, tmp_path, stack_3d):
        # ImageJ-style TIFFs squeeze a single frame to 2D when read back, so we
        # compare after squeezing instead of asserting on the exact shape.
        out = save_tiff_stack(stack_3d, tmp_path, "out", indices=[2])

        result = tifffile.imread(out)

        np.testing.assert_array_equal(np.squeeze(result), stack_3d[2])

    def test_indices_on_4d_stack(self, tmp_path):
        data = np.arange(3 * 2 * 4 * 5, dtype=np.float32).reshape(3, 2, 4, 5)

        out = save_tiff_stack(data, tmp_path, "out", indices=[0, 2], axes="TZYX")

        np.testing.assert_array_equal(tifffile.imread(out), data[[0, 2]])

    def test_indices_on_2d_stack_raises(self, tmp_path, image_2d):
        with pytest.raises(ValueError, match="at least 3 dimensions"):
            save_tiff_stack(image_2d, tmp_path, "out", indices=[0])

    def test_empty_indices_array_raises(self, tmp_path, stack_3d):
        with pytest.raises(ValueError, match="cannot be empty"):
            save_tiff_stack(
                stack_3d, tmp_path, "out", indices=np.array([], dtype=int)
            )

    def test_empty_list_indices_raises(self, tmp_path, stack_3d):
        # np.asarray([]) has dtype float64, so a plain empty list currently
        # fails the integer-dtype check (TypeError) before reaching the
        # "cannot be empty" check. Either way it must not save anything.
        with pytest.raises((TypeError, ValueError)):
            save_tiff_stack(stack_3d, tmp_path, "out", indices=[])

        assert not (tmp_path / "out.tiff").exists()

    @pytest.mark.parametrize(
        "bad_indices",
        [
            [0.0, 1.0],          # floats
            [True, False],       # booleans
            ["0", "1"],          # strings
            [[0, 1], [2, 3]],    # 2D
            1,                   # scalar (0-D)
        ],
    )
    def test_non_integer_or_non_1d_indices_raise_type_error(
        self, tmp_path, stack_3d, bad_indices
    ):
        with pytest.raises(TypeError, match="one-dimensional sequence of integers"):
            save_tiff_stack(stack_3d, tmp_path, "out", indices=bad_indices)

    @pytest.mark.parametrize("bad_indices", [[-1], [0, -3], [5], [0, 99]])
    def test_out_of_range_indices_raise_index_error(
        self, tmp_path, stack_3d, bad_indices
    ):
        # stack_3d has 5 frames, so valid indices are 0..4.
        with pytest.raises(IndexError, match="between 0 and 4"):
            save_tiff_stack(stack_3d, tmp_path, "out", indices=bad_indices)


# --------------------------------------------------------------------------- #
# save_tiff_stack: input validation
# --------------------------------------------------------------------------- #


class TestSaveTiffStackValidation:
    @pytest.mark.parametrize("bad_stack", [[[1, 2], [3, 4]], (1, 2, 3), "image", 5, None])
    def test_non_array_raises_type_error(self, tmp_path, bad_stack):
        with pytest.raises(TypeError, match="numpy array"):
            save_tiff_stack(bad_stack, tmp_path, "out")  # type: ignore[arg-type]

    @pytest.mark.parametrize("shape", [(), (10,)])
    def test_too_few_dimensions_raises(self, tmp_path, shape):
        with pytest.raises(ValueError, match="at least two dimensions"):
            save_tiff_stack(np.zeros(shape, dtype=np.float32), tmp_path, "out")

    @pytest.mark.parametrize("shape", [(0, 4), (3, 0, 4), (0, 4, 4)])
    def test_empty_stack_raises(self, tmp_path, shape):
        with pytest.raises(ValueError, match="empty stack"):
            save_tiff_stack(np.zeros(shape, dtype=np.float32), tmp_path, "out")

    def test_failed_validation_does_not_create_directory_or_file(
        self, tmp_path, stack_3d
    ):
        target = tmp_path / "should_not_exist"

        with pytest.raises(ValueError):
            save_tiff_stack(stack_3d, target, "out", axes="YX")  # wrong axes length

        assert not target.exists()

    def test_failed_index_validation_does_not_create_directory(
        self, tmp_path, stack_3d
    ):
        target = tmp_path / "should_not_exist"

        with pytest.raises(IndexError):
            save_tiff_stack(stack_3d, target, "out", indices=[100])

        assert not target.exists()