"""Tests for bimi.py — BIMI SVG conversion pipeline."""

import re

import pytest
from PIL import Image

from bimi import (
    _detect_format,
    _dominant_color,
    _inline_styles,
    _local_name,
    _ns_tag,
    _parse_css_rules,
    _resolve_use_refs,
    _selector_matches,
    _strip_forbidden_elements,
    _strip_foreign_namespaces,
    convert_to_bimi,
    raster_to_bimi_svg,
    svg_to_bimi_svg,
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
            (".ico", "ico"),
            (".tif", "tiff"),
            (".xyz", "unknown"),
        ],
    )
    def test_extension_mapping(self, ext, expected):
        assert _detect_format(f"/some/path/file{ext}") == expected


class TestDominantColor:
    def test_solid_red(self):
        img = Image.new("RGB", (50, 50), (255, 0, 0))
        assert _dominant_color(img) == "#ff0000"

    def test_solid_white(self):
        img = Image.new("RGB", (50, 50), (255, 255, 255))
        assert _dominant_color(img) == "#ffffff"


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

    def test_has_path_elements(self, white_circle_on_red):
        svg = raster_to_bimi_svg(white_circle_on_red, "Test Corp")
        assert "<path" in svg

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

    def test_ico_input(self, tmp_path):
        """ICO files should be handled via the raster pipeline."""
        img = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        draw.ellipse([16, 16, 48, 48], fill=(0, 0, 255, 255))
        p = tmp_path / "favicon.ico"
        img.save(p, format="ICO")
        svg = convert_to_bimi(str(p), "Fav Co")
        assert "<title>Fav Co</title>" in svg
        assert 'version="1.2"' in svg

    def test_palette_transparency_png(self, tmp_path):
        """Palette-mode PNG with transparency should use alpha-based tracing."""
        img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        draw.ellipse([20, 20, 80, 80], fill=(255, 0, 0, 255))
        # Convert to palette mode, preserving transparency
        img_p = img.convert("P")
        p = tmp_path / "palette.png"
        img_p.save(p, transparency=0)
        svg = convert_to_bimi(str(p), "Palette Co")
        assert "<title>Palette Co</title>" in svg
        assert 'version="1.2"' in svg

    def test_animated_gif(self, tmp_path):
        """Animated GIFs should produce valid output (not crash)."""
        from PIL import ImageDraw

        frames = []
        for color in [(255, 0, 0), (0, 255, 0)]:
            frame = Image.new("RGB", (100, 100), (255, 255, 255))
            draw = ImageDraw.Draw(frame)
            draw.ellipse([20, 20, 80, 80], fill=color)
            frames.append(frame)
        p = tmp_path / "anim.gif"
        frames[0].save(p, save_all=True, append_images=frames[1:], duration=100)
        svg = convert_to_bimi(str(p), "Anim Co")
        assert "<title>Anim Co</title>" in svg
        assert 'version="1.2"' in svg


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


# ---------------------------------------------------------------------------
# CSS parsing & selector matching
# ---------------------------------------------------------------------------


class TestParseCssRules:
    def test_single_class_rule(self):
        css = ".cls-1 { fill: red; stroke: blue; }"
        rules = _parse_css_rules(css)
        assert len(rules) == 1
        sels, props = rules[0]
        assert sels == [".cls-1"]
        assert props == {"fill": "red", "stroke": "blue"}

    def test_comma_separated_selectors(self):
        css = ".a, .b { opacity: 0.5; }"
        rules = _parse_css_rules(css)
        assert len(rules) == 1
        assert rules[0][0] == [".a", ".b"]

    def test_multiple_rules(self):
        css = ".x { fill: red; } circle { stroke: black; }"
        rules = _parse_css_rules(css)
        assert len(rules) == 2

    def test_strips_comments(self):
        css = "/* comment */ .cls { fill: green; }"
        rules = _parse_css_rules(css)
        assert len(rules) == 1
        assert rules[0][1]["fill"] == "green"

    def test_skips_at_rules(self):
        css = '@import url("font.css"); .cls { fill: red; }'
        rules = _parse_css_rules(css)
        assert len(rules) == 1
        assert rules[0][0] == [".cls"]

    def test_font_face_skipped(self):
        css = "@font-face { font-family: MyFont; src: url(f.woff); } .a { fill: red; }"
        rules = _parse_css_rules(css)
        # @font-face selector is filtered out, .a rule kept
        assert any(".a" in r[0] for r in rules)


class TestSelectorMatches:
    def _make_elem(self, tag, cls="", elem_id=""):
        import xml.etree.ElementTree as ET

        e = ET.Element(
            f"{{{_ns_tag.__module__ and 'http://www.w3.org/2000/svg'}}}{tag}"
        )
        if cls:
            e.set("class", cls)
        if elem_id:
            e.set("id", elem_id)
        return e

    def _svg_elem(self, tag, cls="", elem_id=""):
        import xml.etree.ElementTree as ET

        e = ET.Element(_ns_tag(tag))
        if cls:
            e.set("class", cls)
        if elem_id:
            e.set("id", elem_id)
        return e

    def test_type_selector(self):
        e = self._svg_elem("circle")
        assert _selector_matches(e, "circle")
        assert not _selector_matches(e, "rect")

    def test_class_selector(self):
        e = self._svg_elem("path", cls="cls-1 cls-2")
        assert _selector_matches(e, ".cls-1")
        assert _selector_matches(e, ".cls-2")
        assert not _selector_matches(e, ".cls-3")

    def test_id_selector(self):
        e = self._svg_elem("g", elem_id="logo")
        assert _selector_matches(e, "#logo")
        assert not _selector_matches(e, "#other")

    def test_combined_type_class(self):
        e = self._svg_elem("circle", cls="highlight")
        assert _selector_matches(e, "circle.highlight")
        assert not _selector_matches(e, "rect.highlight")

    def test_universal_selector(self):
        e = self._svg_elem("anything")
        assert _selector_matches(e, "*")

    def test_multiple_classes(self):
        e = self._svg_elem("path", cls="a b c")
        assert _selector_matches(e, ".a.b")
        assert not _selector_matches(e, ".a.d")


# ---------------------------------------------------------------------------
# Style inlining
# ---------------------------------------------------------------------------


class TestInlineStyles:
    def test_style_block_to_attrs(self, tmp_path):
        import xml.etree.ElementTree as ET

        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            "<style>.red { fill: red; }</style>"
            '<circle class="red" cx="50" cy="50" r="30"/>'
            "</svg>"
        )
        root = ET.fromstring(svg)
        _inline_styles(root)
        circle = root.find(_ns_tag("circle"))
        assert circle is not None
        assert circle.get("fill") == "red"
        # class should be removed
        assert circle.get("class") is None
        # <style> should be removed
        assert root.find(_ns_tag("style")) is None

    def test_style_attr_overrides_css(self):
        import xml.etree.ElementTree as ET

        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            "<style>.x { fill: red; }</style>"
            '<circle class="x" style="fill: blue;" cx="50" cy="50" r="30"/>'
            "</svg>"
        )
        root = ET.fromstring(svg)
        _inline_styles(root)
        circle = root.find(_ns_tag("circle"))
        assert circle is not None
        assert circle.get("fill") == "blue"

    def test_existing_attr_not_overridden_by_css(self):
        import xml.etree.ElementTree as ET

        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            "<style>.x { fill: red; }</style>"
            '<circle class="x" fill="green" cx="50" cy="50" r="30"/>'
            "</svg>"
        )
        root = ET.fromstring(svg)
        _inline_styles(root)
        circle = root.find(_ns_tag("circle"))
        assert circle is not None
        assert circle.get("fill") == "green"

    def test_style_attr_inlined_without_style_block(self):
        import xml.etree.ElementTree as ET

        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<rect style="fill: orange; stroke: black;" width="10" height="10"/>'
            "</svg>"
        )
        root = ET.fromstring(svg)
        _inline_styles(root)
        rect = root.find(_ns_tag("rect"))
        assert rect is not None
        assert rect.get("fill") == "orange"
        assert rect.get("stroke") == "black"
        assert rect.get("style") is None


# ---------------------------------------------------------------------------
# <use> resolution
# ---------------------------------------------------------------------------


class TestResolveUseRefs:
    def test_basic_use_resolution(self):
        import xml.etree.ElementTree as ET

        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            "<defs>"
            '<circle id="dot" cx="10" cy="10" r="5" fill="red"/>'
            "</defs>"
            '<use href="#dot" x="20" y="30"/>'
            "</svg>"
        )
        root = ET.fromstring(svg)
        _resolve_use_refs(root)
        # <use> should be gone
        assert root.find(f".//{_ns_tag('use')}") is None
        # The circle should be inlined (inside a <g> with translate)
        circles = list(root.iter(_ns_tag("circle")))
        assert len(circles) >= 1

    def test_external_use_removed(self):
        import xml.etree.ElementTree as ET

        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<use href="https://evil.com/logo.svg#mark"/>'
            "</svg>"
        )
        root = ET.fromstring(svg)
        _resolve_use_refs(root)
        assert root.find(f".//{_ns_tag('use')}") is None

    def test_preserves_gradients(self):
        import xml.etree.ElementTree as ET

        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            "<defs>"
            '<linearGradient id="grad1">'
            '<stop offset="0%" stop-color="red"/>'
            '<stop offset="100%" stop-color="blue"/>'
            "</linearGradient>"
            '<circle id="dot" cx="10" cy="10" r="5"/>'
            "</defs>"
            '<use href="#dot"/>'
            '<rect fill="url(#grad1)" width="100" height="100"/>'
            "</svg>"
        )
        root = ET.fromstring(svg)
        _resolve_use_refs(root)
        # Gradient should still exist (referenced by url(#grad1))
        grad = root.find(f".//{_ns_tag('linearGradient')}")
        assert grad is not None
        # Unreferenced circle in defs should be cleaned up
        defs = root.find(_ns_tag("defs"))
        if defs is not None:
            for child in defs:
                assert _local_name(child.tag) != "circle"

    def test_xlink_href(self):
        import xml.etree.ElementTree as ET

        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg"'
            ' xmlns:xlink="http://www.w3.org/1999/xlink">'
            "<defs>"
            '<rect id="box" width="10" height="10" fill="green"/>'
            "</defs>"
            '<use xlink:href="#box"/>'
            "</svg>"
        )
        root = ET.fromstring(svg)
        _resolve_use_refs(root)
        assert root.find(f".//{_ns_tag('use')}") is None
        rects = list(root.iter(_ns_tag("rect")))
        assert len(rects) >= 1


# ---------------------------------------------------------------------------
# Forbidden element / namespace stripping
# ---------------------------------------------------------------------------


class TestStripForbidden:
    def test_strips_script(self):
        import xml.etree.ElementTree as ET

        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            "<script>alert('x')</script>"
            '<circle cx="50" cy="50" r="30"/>'
            "</svg>"
        )
        root = ET.fromstring(svg)
        _strip_forbidden_elements(root)
        assert root.find(_ns_tag("script")) is None
        assert root.find(_ns_tag("circle")) is not None

    def test_strips_clippath(self):
        import xml.etree.ElementTree as ET

        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            "<defs>"
            '<clipPath id="clip"><rect width="50" height="50"/></clipPath>'
            "</defs>"
            '<circle cx="50" cy="50" r="30"/>'
            "</svg>"
        )
        root = ET.fromstring(svg)
        _strip_forbidden_elements(root)
        assert root.find(f".//{_ns_tag('clipPath')}") is None

    def test_strips_filter(self):
        import xml.etree.ElementTree as ET

        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            "<defs>"
            '<filter id="blur"><feGaussianBlur stdDeviation="5"/></filter>'
            "</defs>"
            '<circle cx="50" cy="50" r="30"/>'
            "</svg>"
        )
        root = ET.fromstring(svg)
        _strip_forbidden_elements(root)
        assert root.find(f".//{_ns_tag('filter')}") is None
        assert root.find(f".//{_ns_tag('feGaussianBlur')}") is None


class TestStripForeignNamespaces:
    def test_removes_inkscape_elements(self):
        import xml.etree.ElementTree as ET

        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg"'
            ' xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape">'
            '<inkscape:perspective id="p1"/>'
            '<circle cx="50" cy="50" r="30"/>'
            "</svg>"
        )
        root = ET.fromstring(svg)
        _strip_foreign_namespaces(root)
        # Inkscape element should be gone
        remaining_tags = [_local_name(e.tag) for e in root]
        assert "perspective" not in remaining_tags
        assert root.find(_ns_tag("circle")) is not None


# ---------------------------------------------------------------------------
# End-to-end: SVG with CSS classes (Illustrator-style)
# ---------------------------------------------------------------------------


class TestSvgWithCssClasses:
    def test_illustrator_style_svg(self, tmp_path):
        """Typical Illustrator export with <style> block and class attrs."""
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            "<style>.cls-1{fill:#ff0000;}.cls-2{fill:#00ff00;}</style>"
            '<circle class="cls-1" cx="30" cy="50" r="20"/>'
            '<rect class="cls-2" x="60" y="30" width="30" height="40"/>'
            "</svg>"
        )
        p = tmp_path / "illustrator.svg"
        p.write_text(svg)
        result = svg_to_bimi_svg(str(p), "Test Co")
        # CSS should be inlined
        assert "<style" not in result
        assert 'class="' not in result
        # Colors should be preserved as attributes
        assert "#ff0000" in result
        assert "#00ff00" in result
        # BIMI compliance
        assert 'version="1.2"' in result
        assert 'baseProfile="tiny-ps"' in result
        assert "<title>Test Co</title>" in result

    def test_svg_with_use_element(self, tmp_path):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">'
            "<defs>"
            '<circle id="dot" cx="0" cy="0" r="10" fill="blue"/>'
            "</defs>"
            '<use href="#dot" x="50" y="50"/>'
            '<use href="#dot" x="150" y="50"/>'
            "</svg>"
        )
        p = tmp_path / "uses.svg"
        p.write_text(svg)
        result = svg_to_bimi_svg(str(p), "Dots Inc")
        assert "<use" not in result
        assert "blue" in result
        assert 'baseProfile="tiny-ps"' in result

    def test_svg_with_text_falls_back_to_raster(self, tmp_path):
        """SVGs with <text> should fall back to rasterize-and-retrace."""
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">'
            '<rect width="200" height="100" fill="white"/>'
            '<text x="10" y="50" font-size="20" fill="black">Hello</text>'
            "</svg>"
        )
        p = tmp_path / "text.svg"
        p.write_text(svg)
        result = svg_to_bimi_svg(str(p), "Text Co")
        # Should still produce valid BIMI SVG (via raster fallback)
        assert 'version="1.2"' in result
        assert 'baseProfile="tiny-ps"' in result
        assert "<title>Text Co</title>" in result
        # Text element should NOT appear (rasterized)
        assert "<text" not in result

    def test_already_compliant_svg_passes_through(self, tmp_path):
        """An already-compliant SVG should pass through with minimal changes."""
        svg = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<svg version="1.2" baseProfile="tiny-ps"'
            ' xmlns="http://www.w3.org/2000/svg"'
            ' viewBox="0 0 100 100">'
            "<title>Existing</title>"
            "<desc>Already compliant</desc>"
            '<rect width="100" height="100" fill="white"/>'
            '<circle cx="50" cy="50" r="30" fill="red"/>'
            "</svg>"
        )
        p = tmp_path / "compliant.svg"
        p.write_text(svg)
        result = svg_to_bimi_svg(str(p), "Existing")
        assert 'version="1.2"' in result
        assert 'baseProfile="tiny-ps"' in result
        assert "<circle" in result
        assert "red" in result
