"""Tests for bimi.py — BIMI SVG conversion pipeline."""

import re

import numpy as np
import pytest
from PIL import Image

from bimi import (
    _chord_length_params,
    _compute_path_bounds,
    _detect_format,
    _dominant_color,
    _estimate_center_tangent,
    _estimate_left_tangent,
    _estimate_right_tangent,
    _fit_cubic_beziers,
    _fit_single_cubic,
    _max_bezier_error,
    _trace_with_potrace,
    _upsample_factor,
    convert_to_bimi,
    otsu_threshold,
    raster_to_bimi_svg,
    svg_to_bimi_svg,
    trace_to_svg_paths,
)


# ---------------------------------------------------------------------------
# Fixtures: reusable temp images
# ---------------------------------------------------------------------------


@pytest.fixture
def white_circle_on_red(tmp_path):
    """200x200 PNG: white circle on red background."""
    img = Image.new("RGB", (200, 200), (200, 30, 30))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, 150, 150], fill=(255, 255, 255))
    p = tmp_path / "circle.png"
    img.save(p)
    return str(p)


@pytest.fixture
def black_square_on_white(tmp_path):
    """300x300 JPEG: black square on white background."""
    img = Image.new("RGB", (300, 300), (255, 255, 255))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    draw.rectangle([75, 75, 225, 225], fill=(0, 0, 0))
    p = tmp_path / "square.jpg"
    img.save(p)
    return str(p)


@pytest.fixture
def transparent_logo(tmp_path):
    """100x100 RGBA PNG with a transparent background."""
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 20, 80, 80], fill=(255, 0, 0, 255))
    p = tmp_path / "logo.png"
    img.save(p)
    return str(p)


@pytest.fixture
def simple_svg(tmp_path):
    """A minimal non-BIMI SVG file."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 80">'
        '<circle cx="50" cy="40" r="30" fill="blue"/>'
        "</svg>"
    )
    p = tmp_path / "logo.svg"
    p.write_text(svg)
    return str(p)


# ---------------------------------------------------------------------------
# Otsu thresholding
# ---------------------------------------------------------------------------


class TestOtsuThreshold:
    def test_bimodal_image(self):
        rng = np.random.default_rng(42)
        dark = rng.normal(50, 10, 500).clip(0, 255)
        light = rng.normal(200, 10, 500).clip(0, 255)
        arr = np.concatenate([dark, light]).astype(np.uint8)
        t = otsu_threshold(arr)
        assert 70 < t < 180

    def test_uniform_image(self):
        arr = np.full(1000, 128, dtype=np.uint8)
        t = otsu_threshold(arr)
        assert 0 <= t <= 255

    def test_returns_int(self):
        arr = np.random.randint(0, 256, 1000, dtype=np.uint8)
        assert isinstance(otsu_threshold(arr), int)


# ---------------------------------------------------------------------------
# Format detection & helpers
# ---------------------------------------------------------------------------


class TestDetectFormat:
    @pytest.mark.parametrize(
        "ext,expected",
        [
            (".jpg", "jpeg"),
            (".jpeg", "jpeg"),
            (".png", "png"),
            (".svg", "svg"),
            (".webp", "webp"),
            (".gif", "gif"),
            (".bmp", "bmp"),
            (".tiff", "tiff"),
            (".tif", "tiff"),
            (".xyz", "unknown"),
        ],
    )
    def test_extension_mapping(self, ext, expected):
        assert _detect_format(f"/some/path/file{ext}") == expected


class TestUpsampleFactor:
    def test_small_image(self):
        img = Image.new("RGB", (100, 100))
        assert _upsample_factor(img) == 8

    def test_medium_image(self):
        img = Image.new("RGB", (500, 500))
        assert _upsample_factor(img) == 4

    def test_large_image(self):
        img = Image.new("RGB", (1000, 1000))
        assert _upsample_factor(img) == 2


class TestDominantColor:
    def test_solid_red(self):
        img = Image.new("RGB", (50, 50), (255, 0, 0))
        assert _dominant_color(img) == "#ff0000"

    def test_solid_white(self):
        img = Image.new("RGB", (50, 50), (255, 255, 255))
        assert _dominant_color(img) == "#ffffff"


# ---------------------------------------------------------------------------
# Chord-length parameterization
# ---------------------------------------------------------------------------


class TestChordLengthParams:
    def test_two_points(self):
        pts = np.array([[0.0, 0.0], [10.0, 0.0]])
        params = _chord_length_params(pts)
        np.testing.assert_allclose(params, [0.0, 1.0])

    def test_three_equidistant(self):
        pts = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
        params = _chord_length_params(pts)
        np.testing.assert_allclose(params, [0.0, 0.5, 1.0])

    def test_starts_at_zero_ends_at_one(self):
        pts = np.random.rand(20, 2) * 100
        params = _chord_length_params(pts)
        assert params[0] == pytest.approx(0.0)
        assert params[-1] == pytest.approx(1.0)

    def test_monotonically_increasing(self):
        pts = np.random.rand(20, 2) * 100
        params = _chord_length_params(pts)
        assert np.all(np.diff(params) >= 0)

    def test_coincident_points_fallback(self):
        pts = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
        params = _chord_length_params(pts)
        assert len(params) == 3
        assert params[0] == pytest.approx(0.0)
        assert params[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tangent estimation
# ---------------------------------------------------------------------------


class TestTangentEstimation:
    def test_left_tangent_horizontal(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        t = _estimate_left_tangent(pts)
        np.testing.assert_allclose(t, [1.0, 0.0])

    def test_right_tangent_horizontal(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        t = _estimate_right_tangent(pts)
        np.testing.assert_allclose(t, [-1.0, 0.0])

    def test_tangents_are_unit_vectors(self):
        pts = np.array([[0.0, 0.0], [3.0, 4.0], [10.0, 10.0]])
        assert np.linalg.norm(_estimate_left_tangent(pts)) == pytest.approx(1.0)
        assert np.linalg.norm(_estimate_right_tangent(pts)) == pytest.approx(1.0)

    def test_center_tangent_unit_vector(self):
        pts = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]])
        t = _estimate_center_tangent(pts, 1)
        assert np.linalg.norm(t) == pytest.approx(1.0)

    def test_center_tangent_direction(self):
        pts = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
        t = _estimate_center_tangent(pts, 1)
        np.testing.assert_allclose(t, [1.0, 0.0])


# ---------------------------------------------------------------------------
# Single cubic Bezier fitting
# ---------------------------------------------------------------------------


class TestFitSingleCubic:
    def test_two_points(self):
        pts = np.array([[0.0, 0.0], [9.0, 0.0]])
        params = np.array([0.0, 1.0])
        tl = np.array([1.0, 0.0])
        tr = np.array([-1.0, 0.0])
        cp = _fit_single_cubic(pts, params, tl, tr)
        assert cp.shape == (4, 2)
        np.testing.assert_allclose(cp[0], [0.0, 0.0])
        np.testing.assert_allclose(cp[3], [9.0, 0.0])

    def test_collinear_points(self):
        pts = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0], [9.0, 0.0]])
        params = _chord_length_params(pts)
        tl = np.array([1.0, 0.0])
        tr = np.array([-1.0, 0.0])
        cp = _fit_single_cubic(pts, params, tl, tr)
        # Control points should stay on the x-axis for collinear input
        assert cp[1][1] == pytest.approx(0.0, abs=0.1)
        assert cp[2][1] == pytest.approx(0.0, abs=0.1)

    def test_returns_4_control_points(self):
        pts = np.array([[0.0, 0.0], [2.0, 4.0], [5.0, 3.0], [8.0, 0.0]])
        params = _chord_length_params(pts)
        tl = _estimate_left_tangent(pts)
        tr = _estimate_right_tangent(pts)
        cp = _fit_single_cubic(pts, params, tl, tr)
        assert cp.shape == (4, 2)


# ---------------------------------------------------------------------------
# Bezier error measurement
# ---------------------------------------------------------------------------


class TestMaxBezierError:
    def test_perfect_fit_zero_error(self):
        # Points exactly on a straight line, fit with straight-line Bezier
        cp = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0], [9.0, 0.0]])
        pts = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0], [9.0, 0.0]])
        params = np.array([0.0, 1 / 3, 2 / 3, 1.0])
        err, idx = _max_bezier_error(pts, params, cp)
        assert err == pytest.approx(0.0, abs=1e-6)

    def test_error_increases_with_deviation(self):
        cp = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0], [9.0, 0.0]])
        pts_close = np.array([[0.0, 0.0], [3.0, 0.1], [6.0, 0.0], [9.0, 0.0]])
        pts_far = np.array([[0.0, 0.0], [3.0, 5.0], [6.0, 0.0], [9.0, 0.0]])
        params = np.array([0.0, 1 / 3, 2 / 3, 1.0])
        err_close, _ = _max_bezier_error(pts_close, params, cp)
        err_far, _ = _max_bezier_error(pts_far, params, cp)
        assert err_far > err_close


# ---------------------------------------------------------------------------
# Recursive Bezier fitting
# ---------------------------------------------------------------------------


class TestFitCubicBeziers:
    def _make_semicircle(self, n=50):
        t = np.linspace(0, np.pi, n)
        return np.column_stack([np.sin(t), np.cos(t)]) * 100

    def test_returns_list_of_control_point_arrays(self):
        pts = self._make_semicircle()
        tl = _estimate_left_tangent(pts)
        tr = _estimate_right_tangent(pts)
        result = _fit_cubic_beziers(pts, 1.0, tl, tr)
        assert isinstance(result, list)
        assert all(cp.shape == (4, 2) for cp in result)

    def test_endpoints_match(self):
        pts = self._make_semicircle()
        tl = _estimate_left_tangent(pts)
        tr = _estimate_right_tangent(pts)
        result = _fit_cubic_beziers(pts, 1.0, tl, tr)
        np.testing.assert_allclose(result[0][0], pts[0], atol=1e-6)
        np.testing.assert_allclose(result[-1][3], pts[-1], atol=1e-6)

    def test_tighter_tolerance_more_segments(self):
        pts = self._make_semicircle()
        tl = _estimate_left_tangent(pts)
        tr = _estimate_right_tangent(pts)
        loose = _fit_cubic_beziers(pts, 10.0, tl, tr)
        tight = _fit_cubic_beziers(pts, 0.1, tl, tr)
        assert len(tight) >= len(loose)

    def test_straight_line_single_segment(self):
        pts = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0], [9.0, 0.0]])
        tl = _estimate_left_tangent(pts)
        tr = _estimate_right_tangent(pts)
        result = _fit_cubic_beziers(pts, 1.0, tl, tr)
        assert len(result) == 1

    def test_two_points(self):
        pts = np.array([[0.0, 0.0], [10.0, 0.0]])
        tl = np.array([1.0, 0.0])
        tr = np.array([-1.0, 0.0])
        result = _fit_cubic_beziers(pts, 1.0, tl, tr)
        assert len(result) == 1
        assert result[0].shape == (4, 2)


# ---------------------------------------------------------------------------
# trace_to_svg_paths
# ---------------------------------------------------------------------------


class TestTraceToSvgPaths:
    def _circle_mask(self, size=200):
        y, x = np.ogrid[:size, :size]
        center = size // 2
        return ((x - center) ** 2 + (y - center) ** 2) < (size // 4) ** 2

    def test_produces_cubic_commands(self):
        mask = self._circle_mask()
        paths = trace_to_svg_paths(mask, 200, tolerance=2.0)
        assert len(paths) >= 1
        for p in paths:
            assert p.startswith("M")
            assert "C" in p
            assert p.endswith("Z")

    def test_no_line_commands(self):
        mask = self._circle_mask()
        paths = trace_to_svg_paths(mask, 200, tolerance=2.0)
        for p in paths:
            # Should have no L commands — only M, C, Z
            assert "L" not in p

    def test_empty_mask_no_paths(self):
        mask = np.zeros((100, 100), dtype=bool)
        paths = trace_to_svg_paths(mask, 100)
        assert paths == []

    def test_full_mask_produces_boundary_path(self):
        mask = np.ones((100, 100), dtype=bool)
        paths = trace_to_svg_paths(mask, 100)
        # Padding creates a contour at the mask boundary
        assert len(paths) == 1
        assert paths[0].startswith("M")
        assert "C" in paths[0]


# ---------------------------------------------------------------------------
# Path bounds
# ---------------------------------------------------------------------------


class TestComputePathBounds:
    def test_simple_path(self):
        paths = ["M10.0,20.0C15.0,25.0 30.0,35.0 40.0,50.0Z"]
        min_x, min_y, max_x, max_y = _compute_path_bounds(paths)
        assert min_x == pytest.approx(10.0)
        assert min_y == pytest.approx(20.0)
        assert max_x == pytest.approx(40.0)
        assert max_y == pytest.approx(50.0)

    def test_multiple_paths(self):
        paths = [
            "M0.0,0.0C5.0,5.0 10.0,10.0 10.0,10.0Z",
            "M20.0,20.0C25.0,25.0 30.0,30.0 30.0,30.0Z",
        ]
        min_x, min_y, max_x, max_y = _compute_path_bounds(paths)
        assert min_x == pytest.approx(0.0)
        assert min_y == pytest.approx(0.0)
        assert max_x == pytest.approx(30.0)
        assert max_y == pytest.approx(30.0)

    def test_negative_coords(self):
        paths = ["M-5.0,-10.0C0.0,0.0 5.0,10.0 5.0,10.0Z"]
        min_x, min_y, max_x, max_y = _compute_path_bounds(paths)
        assert min_x == pytest.approx(-5.0)
        assert min_y == pytest.approx(-10.0)


# ---------------------------------------------------------------------------
# End-to-end: raster_to_bimi_svg
# ---------------------------------------------------------------------------


class TestRasterToBimiSvg:
    def test_produces_valid_svg(self, white_circle_on_red):
        svg = raster_to_bimi_svg(white_circle_on_red, "Test Corp")
        assert svg.startswith("<svg")
        assert "</svg>" in svg

    def test_contains_bimi_attributes(self, white_circle_on_red):
        svg = raster_to_bimi_svg(white_circle_on_red, "Test Corp")
        assert 'version="1.2"' in svg
        assert 'baseProfile="tiny-ps"' in svg

    def test_square_viewbox(self, white_circle_on_red):
        svg = raster_to_bimi_svg(white_circle_on_red, "Test Corp")
        vb = re.search(r'viewBox="0 0 (\d+) (\d+)"', svg)
        assert vb
        assert vb.group(1) == vb.group(2)

    def test_has_title_and_desc(self, white_circle_on_red):
        svg = raster_to_bimi_svg(white_circle_on_red, "Test Corp")
        assert "<title>Test Corp</title>" in svg
        assert "<desc>Test Corp logo mark</desc>" in svg

    def test_has_background_rect(self, white_circle_on_red):
        svg = raster_to_bimi_svg(white_circle_on_red, "Test Corp")
        assert "<rect" in svg

    def test_uses_bezier_curves(self, white_circle_on_red):
        svg = raster_to_bimi_svg(white_circle_on_red, "Test Corp")
        # Paths should contain C commands
        path_d = re.findall(r'd="([^"]*)"', svg)
        assert any("C" in d for d in path_d)

    def test_under_32kb(self, white_circle_on_red):
        svg = raster_to_bimi_svg(white_circle_on_red, "Test Corp")
        assert len(svg.encode("utf-8")) <= 32 * 1024

    def test_jpeg_input(self, black_square_on_white):
        svg = raster_to_bimi_svg(black_square_on_white, "Square Inc")
        assert "<title>Square Inc</title>" in svg
        assert 'version="1.2"' in svg

    def test_transparent_png(self, transparent_logo):
        svg = raster_to_bimi_svg(transparent_logo, "Alpha Co")
        assert "<title>Alpha Co</title>" in svg


# ---------------------------------------------------------------------------
# Potrace integration
# ---------------------------------------------------------------------------


class TestPotrace:
    def test_returns_none_when_unavailable(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        mask = np.ones((50, 50), dtype=bool)
        assert _trace_with_potrace(mask) is None

    def test_returns_paths_and_transform_when_available(self, white_circle_on_red):
        """Only runs where potrace is installed (Docker)."""
        import shutil

        if not shutil.which("potrace"):
            pytest.skip("potrace not installed")
        from bimi import preprocess_raster

        binary, _, _ = preprocess_raster(white_circle_on_red)
        result = _trace_with_potrace(binary)
        assert result is not None
        paths, xform = result
        assert len(paths) >= 1
        assert paths[0].startswith("M") or paths[0].startswith("m")
        assert "scale" in xform


# ---------------------------------------------------------------------------
# End-to-end: svg_to_bimi_svg
# ---------------------------------------------------------------------------


class TestSvgToBimiSvg:
    def test_adds_bimi_attributes(self, simple_svg):
        svg = svg_to_bimi_svg(simple_svg, "Blue Dot")
        assert 'version="1.2"' in svg
        assert 'baseProfile="tiny-ps"' in svg

    def test_makes_viewbox_square(self, simple_svg):
        svg = svg_to_bimi_svg(simple_svg, "Blue Dot")
        vb = re.search(r'viewBox="[+-]?[\d.]+ [+-]?[\d.]+ ([\d.]+) ([\d.]+)"', svg)
        assert vb
        assert float(vb.group(1)) == pytest.approx(float(vb.group(2)))

    def test_adds_title_and_desc(self, simple_svg):
        svg = svg_to_bimi_svg(simple_svg, "Blue Dot")
        assert "<title>Blue Dot</title>" in svg
        assert "<desc>Blue Dot logo mark</desc>" in svg

    def test_adds_background_rect(self, simple_svg):
        svg = svg_to_bimi_svg(simple_svg, "Blue Dot")
        assert "<rect" in svg

    def test_strips_forbidden_elements(self, tmp_path):
        svg_with_script = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            '<script>alert("x")</script>'
            '<circle cx="50" cy="50" r="30" fill="red"/>'
            "</svg>"
        )
        p = tmp_path / "bad.svg"
        p.write_text(svg_with_script)
        result = svg_to_bimi_svg(str(p), "Test")
        assert "<script" not in result

    def test_strips_external_hrefs(self, tmp_path):
        svg_with_href = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            '<use href="https://evil.com/exploit.svg"/>'
            "</svg>"
        )
        p = tmp_path / "ext.svg"
        p.write_text(svg_with_href)
        result = svg_to_bimi_svg(str(p), "Test")
        assert "https://" not in result

    def test_has_xml_declaration(self, simple_svg):
        svg = svg_to_bimi_svg(simple_svg, "Blue Dot")
        assert svg.startswith("<?xml")


# ---------------------------------------------------------------------------
# End-to-end: convert_to_bimi (public API)
# ---------------------------------------------------------------------------


class TestConvertToBimi:
    def test_png_input(self, white_circle_on_red):
        svg = convert_to_bimi(white_circle_on_red, "Acme")
        assert "<title>Acme</title>" in svg

    def test_svg_input(self, simple_svg):
        svg = convert_to_bimi(simple_svg, "Acme")
        assert 'baseProfile="tiny-ps"' in svg

    def test_unknown_format_raises(self, tmp_path):
        p = tmp_path / "file.xyz"
        p.write_bytes(b"not an image")
        with pytest.raises(ValueError, match="Unsupported"):
            convert_to_bimi(str(p), "Nope")

    def test_oversized_svg_raises(self, tmp_path):
        # Create an SVG that will exceed 32KB after conversion
        huge_path = "M0,0" + " L1,1" * 10000
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            f'<path d="{huge_path}"/>'
            "</svg>"
        )
        p = tmp_path / "huge.svg"
        p.write_text(svg)
        with pytest.raises(ValueError, match="32 KB"):
            convert_to_bimi(str(p), "Big")
